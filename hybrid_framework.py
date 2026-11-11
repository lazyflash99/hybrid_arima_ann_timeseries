"""
GenericHybrid(linear_model, nonlinear_model, combiner): every component has fit(y_train) / predict_one_step(y_all).

Linear:     ARIMA, SARIMA, ETS, Theta, OLS AR, Naive, Seasonal naive
Nonlinear:  MLP (the Eq. 2 network), SVR, LightGBM (RandomForest if LightGBM is missing), GRU (skipped without torch)
Combiners:  additive (Zhang), multiplicative, joint-input (Extension style), validation-weighted

Usage:  python hybrid_framework.py [--quick | --plots-only]
Writes results/4_alternative_hybrids_grid/ and figures/4_alternative_hybrids_grid/.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import itertools
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import arima_model as am
from ann_model import NeuralNetwork, lag_matrix
from config import MODES, VAL_FRAC, WORKERS, figure_path, results_path
from data_loader import DATASETS
from validation import diebold_mariano, metrics, refit_origins, validation_folds

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None
try:
    import torch
except ImportError:
    torch = None


# ------------------------------------------------------------------------------- linear models
class Naive:
    max_lag = 1

    def fit(self, y):
        return self

    def predict_one_step(self, y):
        return np.r_[np.nan, np.asarray(y, float)[:-1]]


class SeasonalNaive:
    def __init__(self, m):
        self.max_lag = m if m > 1 else 1

    def fit(self, y):
        return self

    def predict_one_step(self, y):
        y = np.asarray(y, float)
        return np.r_[np.full(self.max_lag, np.nan), y[:-self.max_lag]]


class OLSAR:
    def __init__(self, p=12):
        self.p = self.max_lag = p

    def fit(self, y):
        self.p = self.max_lag = min(self.p, max(1, len(y) // 5))
        X, t = lag_matrix(np.asarray(y, float), self.p, self.p)
        self.b = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), t, rcond=None)[0]
        return self

    def predict_one_step(self, y):
        X, _ = lag_matrix(np.asarray(y, float), self.p, self.p)
        return np.r_[np.full(self.p, np.nan), self.b[0] + X @ self.b[1:]]


class SARIMA:
    """SARIMA(1,d,1)(0,D,1,m), d and D from the training data; non-seasonal if m is 1 or above 24."""

    def __init__(self, m):
        self.m = m if 1 < m <= 24 else 0

    def fit(self, y):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from stationarity import seasonal_strength
        d = am.choose_d(y, max_d=1)
        D = int(self.m > 0 and seasonal_strength(y, self.m) >= 0.64)
        self.order = (1, d, 1)
        self.seasonal = (0, D, 1, self.m) if self.m else (0, 0, 0, 0)
        self.max_lag = d + D * self.m + 1
        self.params = SARIMAX(y, order=self.order, seasonal_order=self.seasonal, trend="n" if d + D else "c").fit(disp=False, maxiter=60).params
        self.trend = "n" if d + D else "c"
        return self

    def predict_one_step(self, y):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        res = SARIMAX(np.asarray(y, float), order=self.order, seasonal_order=self.seasonal, trend=self.trend).smooth(self.params)
        f = np.asarray(res.get_prediction(0, len(y) - 1).predicted_mean, float)
        f[:self.max_lag] = np.nan
        return f


class ETS:
    """Holt-Winters with additive trend and (for 1 < m <= 24) additive season."""

    def __init__(self, m):
        self.m = m if 1 < m <= 24 else 0
        self.max_lag = max(self.m, 2)

    def _model(self, y):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        return ExponentialSmoothing(np.asarray(y, float), trend="add", seasonal="add" if self.m else None,
                                    seasonal_periods=self.m or None, initialization_method="heuristic" if self.m else "estimated")

    def fit(self, y):
        r = self._model(y).fit()
        self.p = {k: r.params[k] for k in ("smoothing_level", "smoothing_trend", "smoothing_seasonal") if r.params.get(k) is not None and np.isfinite(r.params[k])}
        return self

    def predict_one_step(self, y):
        r = self._model(y).fit(optimized=False, **self.p)
        f = np.asarray(r.fittedvalues, float)
        f[:self.max_lag] = np.nan
        return f


class Theta:
    """Theta method: SES on the series plus half of the training trend slope; additive de-seasonalising if m is 2-24."""

    def __init__(self, m):
        self.m = m if 1 < m <= 24 else 0
        self.max_lag = max(self.m, 2)

    def _ses(self, y, a):
        lev, out = y[0], np.full(len(y), np.nan)
        for t in range(1, len(y)):
            out[t] = lev
            lev = a * y[t] + (1 - a) * lev
        return out

    def _season(self, n):
        return self.s[np.arange(n) % self.m] if self.m else np.zeros(n)

    def fit(self, y):
        y = np.asarray(y, float)
        if self.m:
            self.s = np.array([np.mean(y[i::self.m]) for i in range(self.m)]) - y.mean()
        z = y - self._season(len(y))
        self.slope = np.polyfit(np.arange(len(z)), z, 1)[0]
        grid = np.linspace(0.05, 0.95, 19)
        self.a = grid[int(np.argmin([np.nanmean((z - self._ses(z, a)) ** 2) for a in grid]))]
        return self

    def predict_one_step(self, y):
        y = np.asarray(y, float)
        s = self._season(len(y))
        f = self._ses(y - s, self.a) + 0.5 * self.slope + s
        f[:self.max_lag] = np.nan
        return f


def make_linear(name, m):
    return dict(ARIMA=lambda: am.AutoARIMA(verbose=False), SARIMA=lambda: SARIMA(m), ETS=lambda: ETS(m), Theta=lambda: Theta(m),
                Naive=Naive)[name]() if name in ("ARIMA", "SARIMA", "ETS", "Theta", "Naive") else \
        (OLSAR() if name == "OLS AR" else SeasonalNaive(m))


# ------------------------------------------------------------------------------- nonlinear models
class LagRegressor:
    """Any scikit-learn style regressor on the last p values of the series, scaled with the training range."""

    def __init__(self, est, p):
        self.est, self.p = est, p

    def fit(self, s):
        self.lo, self.rng = np.min(s), max(np.ptp(s), 1e-8)
        X, y = lag_matrix((np.asarray(s) - self.lo) / self.rng, self.p, self.p)
        self.est.fit(X, y)
        return self

    def predict_one_step(self, s):
        X, _ = lag_matrix((np.asarray(s) - self.lo) / self.rng, self.p, self.p)
        return np.r_[np.full(self.p, np.nan), self.est.predict(X) * self.rng + self.lo]


class SmallGRU:
    """One GRU layer of 8 units on the last p values, CPU-friendly, Adam, 150 full-batch steps."""

    def __init__(self, p, seed):
        self.p, self.seed = p, seed

    def _x(self, s):
        return torch.tensor(lag_matrix((np.asarray(s) - self.lo) / self.rng, self.p, self.p)[0][:, ::-1].copy(), dtype=torch.float32).unsqueeze(-1)

    def fit(self, s):
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        self.lo, self.rng = np.min(s), max(np.ptp(s), 1e-8)
        y = torch.tensor(lag_matrix((np.asarray(s) - self.lo) / self.rng, self.p, self.p)[1], dtype=torch.float32).unsqueeze(-1)
        self.gru, self.out = torch.nn.GRU(1, 8, batch_first=True), torch.nn.Linear(8, 1)
        opt = torch.optim.Adam(list(self.gru.parameters()) + list(self.out.parameters()), lr=0.01)
        X = self._x(s)
        for _ in range(150):
            opt.zero_grad()
            loss = torch.mean((self.out(self.gru(X)[0][:, -1]) - y) ** 2)
            loss.backward(); opt.step()
        return self

    def predict_one_step(self, s):
        with torch.no_grad():
            p = self.out(self.gru(self._x(s))[0][:, -1]).numpy().ravel()
        return np.r_[np.full(self.p, np.nan), p * self.rng + self.lo]


def available_nonlinear(names):
    return [n for n in names if not (n == "GRU" and torch is None)]


def make_nonlinear(name, p, seed):
    if name == "MLP":
        return NeuralNetwork(p, 4, seed=seed)
    if name == "SVR":
        return LagRegressor(SVR(kernel="rbf", C=1.0, epsilon=0.01), p)
    if name == "LightGBM":
        est = LGBMRegressor(n_estimators=100, learning_rate=0.05, num_leaves=7, min_child_samples=5, random_state=seed, verbose=-1, n_jobs=1) \
            if LGBMRegressor else RandomForestRegressor(n_estimators=100, min_samples_leaf=3, random_state=seed, n_jobs=1)
        return LagRegressor(est, p)
    return SmallGRU(p, seed)


# ------------------------------------------------------------------------------- the generic hybrid
class GenericHybrid:
    """combiner: 'additive' y = L + N(e); 'multiplicative' y = L * N(y / L); 'joint-input' network on y lags, L, e lags;
    'validation-weighted' y = L + w N(e), w in [0, 1] chosen on the last part of the training data."""

    def __init__(self, make_lin, make_nl, combiner, p, seed=0):
        self.make_lin, self.make_nl, self.combiner, self.p, self.seed = make_lin, make_nl, combiner, p, seed

    def _ratio_ok(self, L):
        return np.where(np.abs(L) < self.floor, np.sign(L + 1e-12) * self.floor, L)

    def _fit_parts(self, y):
        self.lin = self.make_lin().fit(y)
        L = self.lin.predict_one_step(y)
        self.first = int(self.lin.max_lag)
        if self.combiner == "multiplicative":
            self.floor = max(1e-3, 0.05 * np.nanmean(np.abs(y)))
            self.nl = self.make_nl().fit(np.clip((y / self._ratio_ok(L))[self.first:], 0.2, 5.0))
        elif self.combiner == "joint-input":
            self._fit_joint(y, L)
        else:
            self.nl = self.make_nl().fit((y - L)[self.first:])

    def _joint_x(self, y, L):
        s, e = (y - self.lo) / self.rng, (y - L) / self.rng
        t = np.arange(self.start, len(y))
        return np.column_stack([s[t - k] for k in range(1, self.p + 1)] + [(L[t] - self.lo) / self.rng] +
                               [e[t - k] for k in range(1, 3)]), t

    def _fit_joint(self, y, L):
        self.lo, self.rng = np.min(y), max(np.ptp(y), 1e-8)
        self.start = max(self.p, self.first + 2)
        X, t = self._joint_x(y, L)
        base = self.make_nl()
        if isinstance(base, NeuralNetwork):
            base.n_inputs = X.shape[1]
            self.net = NeuralNetwork(X.shape[1], base.q, seed=self.seed)
            self.w = self.net._train(X, (y[t] - self.lo) / self.rng)
        elif isinstance(base, LagRegressor):
            self.net = base.est.fit(X, (y[t] - self.lo) / self.rng)
        else:
            raise TypeError("joint-input is not defined for this nonlinear model")

    def _raw(self, y):
        y = np.asarray(y, float)
        L = self.lin.predict_one_step(y)
        out = np.full(len(y), np.nan)
        if self.combiner == "multiplicative":
            r = np.clip(y / self._ratio_ok(L), 0.2, 5.0)
            n = np.full(len(y), np.nan); n[self.first:] = self.nl.predict_one_step(r[self.first:])
            return L, L * n
        if self.combiner == "joint-input":
            X, t = self._joint_x(y, L)
            p = self.net._forward(self.w, X)[1] if isinstance(self.net, NeuralNetwork) else self.net.predict(X)
            out[t] = p * self.rng + self.lo
            return L, out
        n = np.full(len(y), np.nan); n[self.first:] = self.nl.predict_one_step((y - L)[self.first:])
        return L, L + n

    def fit(self, y):
        y = np.asarray(y, float)
        self.w_val = 1.0
        if self.combiner == "validation-weighted":
            n_fit = int(round(len(y) * (1 - VAL_FRAC)))
            self._fit_parts(y[:n_fit])
            L, full = self._raw(y)
            ok = n_fit + np.where(~np.isnan(full[n_fit:]))[0]
            grid = np.linspace(0, 1, 11)
            self.w_val = grid[int(np.argmin([np.mean((y[ok] - (L[ok] + w * (full - L)[ok])) ** 2) for w in grid]))]
        self._fit_parts(y)
        return self

    def predict_one_step(self, y):
        L, f = self._raw(y)
        return L + self.w_val * (f - L) if self.combiner == "validation-weighted" else f


# ------------------------------------------------------------------------------- the grid
def make_combo(lin, nl, comb, p, seed, m):
    return GenericHybrid(lambda: make_linear(lin, m), lambda: make_nonlinear(nl, p, seed), comb, p, seed)


def valid_combo(nl, comb):
    return not (nl == "GRU" and comb == "joint-input")


def tune(job):
    """Validation folds inside the training part only; every combination tries the same lags grid."""
    name, lin, nl, comb, mode = job
    warnings.filterwarnings("ignore")
    cfg, d = MODES[mode], DATASETS[name]
    am.DEFAULT_SUBSET = False
    y = d["load"]()[:d["n_train"]]
    folds = validation_folds(len(y), 1)
    best = (np.inf, None)
    for p in cfg["fw_grid"]["lags"]:
        try:
            a, b = folds[0]
            m = make_combo(lin, nl, comb, p, 0, d["period"]).fit(y[:a])
            f = m.predict_one_step(y[:b])[a:b]
            ok = np.isfinite(f)
            if ok.mean() < 0.9:
                continue
            score = float(np.mean((y[a:b][ok] - f[ok]) ** 2))
        except Exception:
            continue
        if score < best[0]:
            best = (score, p)
    return name, lin, nl, comb, best[1], best[0]


def final(job):
    name, lin, nl, comb, p, seed, mode = job
    warnings.filterwarnings("ignore")
    cfg, d = MODES[mode], DATASETS[name]
    am.DEFAULT_SUBSET = False
    y, n = d["load"](), d["n_train"]
    head = dict(dataset=name, linear=lin, nonlinear=nl, combiner=comb, lags=p, seed=seed)
    try:
        origins = refit_origins(n, len(y), min(2, cfg["n_refits"]))
        f = np.full(len(y) - n, np.nan)
        for i, s in enumerate(origins):
            e = origins[i + 1] if i + 1 < len(origins) else len(y)
            f[s - n:e - n] = make_combo(lin, nl, comb, p, seed, d["period"]).fit(y[:s]).predict_one_step(y[:e])[s:e]
        if not np.isfinite(f).all():
            raise ValueError("non-finite")
    except Exception as ex:
        return dict(head, status=f"failed: {type(ex).__name__}"), None
    return dict(head, status="ok", **metrics(y[n:], f, y[:n])), f


def single(job):
    """Single-model baselines under the same protocol: each linear model alone, each nonlinear model alone."""
    name, kind, label, mode = job
    warnings.filterwarnings("ignore")
    cfg, d = MODES[mode], DATASETS[name]
    am.DEFAULT_SUBSET = False
    y, n = d["load"](), d["n_train"]
    try:
        if kind == "linear":
            make = lambda: make_linear(label, d["period"])
        else:
            make = lambda: make_nonlinear(label, cfg["fw_grid"]["lags"][-1], 0)
        origins = refit_origins(n, len(y), min(2, cfg["n_refits"]))
        f = np.full(len(y) - n, np.nan)
        for i, s in enumerate(origins):
            e = origins[i + 1] if i + 1 < len(origins) else len(y)
            f[s - n:e - n] = make().fit(y[:s]).predict_one_step(y[:e])[s:e]
        if not np.isfinite(f).all():
            raise ValueError
        return dict(dataset=name, kind=kind, model=label, **metrics(y[n:], f, y[:n])), f
    except Exception:
        return dict(dataset=name, kind=kind, model=label, MSE=np.nan), None


COMBINER_ORDER = ["additive", "multiplicative", "joint-input", "validation-weighted"]


def plot_heatmaps(names):
    """One heatmap per dataset from the saved grid results: test MSE of every design divided by the MSE of the
    best single model. The panels share one column of row labels so that the labels cannot overlap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df, singles = pd.read_csv(results_path("grid_results.csv")), pd.read_csv(results_path("grid_single_models.csv"))
    mean = df[df.status == "ok"].groupby(["dataset", "linear", "nonlinear", "combiner"], as_index=False)["MSE"].mean()
    for name in names:
        g = mean[mean.dataset == name]
        s = singles[(singles.dataset == name) & singles.MSE.notna()]
        if g.empty or s.empty:
            continue
        best = s.sort_values("MSE").iloc[0]
        combs = [c for c in COMBINER_ORDER if c in set(g.combiner)]
        lins, nls = sorted(g.linear.unique()), sorted(g.nonlinear.unique())
        vals = g.MSE / best["MSE"]
        vmax = min(np.nanmax(vals), max(2.0, np.nanmin(vals) * 2))
        fig, axes = plt.subplots(1, len(combs), figsize=(3.0 * len(combs) + 2.2, 3.9), sharey=True, squeeze=False,
                                 constrained_layout=True)
        for k, (ax, comb) in enumerate(zip(axes[0], combs)):
            piv = g[g.combiner == comb].assign(rel=lambda x: x.MSE / best["MSE"]).pivot(
                index="linear", columns="nonlinear", values="rel").reindex(index=lins, columns=nls)
            im = ax.imshow(piv.values, cmap="RdBu_r", vmin=2 - vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(nls)))
            ax.set_xticklabels(nls, fontsize=8, rotation=30, ha="right")
            ax.set_yticks(range(len(lins)))
            if k == 0:
                ax.set_yticklabels(lins, fontsize=8)
            ax.set_title(comb, fontsize=10)
            for i, j in itertools.product(range(len(lins)), range(len(nls))):
                if np.isfinite(piv.values[i, j]):
                    ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.suptitle(f"{name}: test MSE relative to the best single model ({best['model']}); below 1 is better", fontsize=10)
        fig.colorbar(im, ax=axes[0].tolist(), shrink=.85, label="test MSE / best single model")
        fig.savefig(figure_path("grid", name.replace(" ", "_").lower() + ".png"), dpi=110)
        plt.close(fig)


def report(df, singles, forecasts, single_f, names):
    ok = df[df.status == "ok"]
    mean = ok.groupby(["dataset", "linear", "nonlinear", "combiner"], as_index=False)["MSE"].mean()
    leaders, when = [], []
    stat = pd.read_csv(results_path("stationarity.csv")).set_index("dataset") if os.path.exists(results_path("stationarity.csv")) else None
    nl_tab = pd.read_csv(results_path("nonlinearity_vs_improvement.csv")).set_index("dataset") if os.path.exists(results_path("nonlinearity_vs_improvement.csv")) else None
    for name in names:
        d = DATASETS[name]
        a = d["load"]()[d["n_train"]:]
        g = mean[mean.dataset == name]
        s = singles[(singles.dataset == name) & singles.MSE.notna()]
        if g.empty or s.empty:
            continue
        best_single = s.sort_values("MSE").iloc[0]
        ref_err = a - single_f[(name, best_single["model"])]
        rows = g.sort_values("MSE").head(10)
        for rank, r in enumerate(rows.itertuples(), 1):
            key = (name, r.linear, r.nonlinear, r.combiner)
            dm, p = diebold_mariano(ref_err, a - np.mean(forecasts[key], axis=0))
            leaders.append(dict(dataset=name, rank=rank, linear=r.linear, nonlinear=r.nonlinear, combiner=r.combiner, MSE=r.MSE,
                                relative_to_best_single=r.MSE / best_single["MSE"], best_single=best_single["model"], dm_stat=dm, dm_p=p))
        top = g.sort_values("MSE").iloc[0]
        when.append(dict(dataset=name, best_hybrid=f"{top.linear} + {top.nonlinear} ({top.combiner})", best_hybrid_mse=top.MSE,
                         best_single=best_single["model"], best_single_mse=best_single["MSE"], simple_baseline_wins=bool(best_single["MSE"] <= top.MSE),
                         seasonal_strength=stat.loc[name, "seasonal_strength"] if stat is not None and name in stat.index else np.nan,
                         nonlinearity_detected=nl_tab.loc[name, "nonlinearity_detected"] if nl_tab is not None and name in nl_tab.index else None))
    pd.DataFrame(leaders).to_csv(results_path("grid_leaderboard.csv"), index=False)
    w = pd.DataFrame(when)
    w.to_csv(results_path("grid_when.csv"), index=False)
    plot_heatmaps(names)
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 50):
        print(w.round(5).to_string(index=False))


def main(mode="full"):
    cfg = MODES[mode]
    names = cfg["datasets"] or list(DATASETS)
    nls = available_nonlinear(cfg["fw_nl"])
    t0 = time.time()
    combos = [(n, l, nl, c, mode) for n in names for l in cfg["fw_lin"] for nl in nls for c in cfg["fw_comb"] if valid_combo(nl, c)]
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        tuned = list(pool.map(tune, combos, chunksize=2))
        print(f"tuned {len(tuned)} combinations in {time.time() - t0:.0f} s", flush=True)
        singles = list(pool.map(single, [(n, k, m, mode) for n in names for k, ms in (("linear", cfg["fw_lin"]), ("nonlinear", nls)) for m in ms]))
        jobs = [(n, l, nl, c, p, s, mode) for n, l, nl, c, p, _ in tuned if p is not None for s in (cfg["fw_seeds"] if nl in ("MLP", "GRU", "LightGBM") else cfg["fw_seeds"][:1])]
        results = list(pool.map(final, jobs, chunksize=4))
    rows, fc = [], {}
    for r, f in results:
        rows.append(r)
        if f is not None:
            fc.setdefault((r["dataset"], r["linear"], r["nonlinear"], r["combiner"]), []).append(f)
    df = pd.DataFrame(rows)
    df.to_csv(results_path("grid_results.csv"), index=False)
    sdf = pd.DataFrame([r for r, _ in singles])
    sdf.to_csv(results_path("grid_single_models.csv"), index=False)
    single_f = {(r["dataset"], r["model"]): f for r, f in singles if f is not None}
    no = [t for t in tuned if t[4] is None]
    print(f"{len(df)} runs, {int((df.status == 'ok').sum())} ok; combinations with no valid validation fit: {len(no)}; {time.time() - t0:.0f} s", flush=True)
    report(df, sdf, fc, single_f, names)


if __name__ == "__main__":
    if "--plots-only" in sys.argv:            # redraw the heatmaps from the saved results, without rerunning the grid
        plot_heatmaps(list(DATASETS))
    else:
        main("quick" if "--quick" in sys.argv else "full")
