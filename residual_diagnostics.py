"""
Residual diagnostics for ARIMA residuals (train and test) and hybrid residuals (test), fixed protocol, seed 0.
Usage:  python residual_diagnostics.py [--quick]
Writes results/3_residuals_and_decomposition/, figures/3_residual_diagnostics/*.png and
RESIDUAL_FINDINGS.md (text generated from the test results).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import bds

import arima_model as am
from config import MODES, ROOT, WORKERS, figure_path, results_path
from data_loader import DATASETS
from validation import build, registry_params

ALPHA = 0.05


def teraesvirta_test(e, lags=2):
    """Neural-network linearity test: F test of second and third order terms of lagged residuals."""
    y = e[lags:]
    X = np.column_stack([e[lags - k:-k] for k in range(1, lags + 1)])
    X0 = np.column_stack([np.ones(len(y)), X])
    r0 = y - X0 @ np.linalg.lstsq(X0, y, rcond=None)[0]
    terms = [X[:, i] * X[:, j] for i in range(lags) for j in range(i, lags)] + \
            [X[:, i] * X[:, j] * X[:, k] for i in range(lags) for j in range(i, lags) for k in range(j, lags)]
    X1 = np.column_stack([X0] + terms)
    r1 = y - X1 @ np.linalg.lstsq(X1, y, rcond=None)[0]
    df1, df2 = len(terms), len(y) - X1.shape[1]
    F = ((r0 @ r0 - r1 @ r1) / df1) / (r1 @ r1 / df2)
    return float(F), float(stats.f.sf(F, df1, df2))


def tests(e, level):
    n = len(e)
    lb_lags = [k for k in (6, 12) if k < n / 2]
    q = acorr_ljungbox(e, lags=lb_lags, boxpierce=True)
    out = {f"ljung_box_p_{k}": float(q.loc[k, "lb_pvalue"]) for k in lb_lags}
    out.update({f"box_pierce_p_{k}": float(q.loc[k, "bp_pvalue"]) for k in lb_lags})
    out["jarque_bera_p"] = float(stats.jarque_bera(e).pvalue)
    out["shapiro_p"] = float(stats.shapiro(e[:5000]).pvalue)
    out["arch_lm_p"] = float(het_arch(e, nlags=min(4, n // 5))[1])
    out["bds_p"] = float(np.atleast_1d(bds(e, max_dim=2)[1])[0]) if n >= 30 else np.nan
    out["nn_linearity_p"] = teraesvirta_test(e)[1]
    out["var_vs_level_spearman"], out["var_vs_level_p"] = map(float, stats.spearmanr(np.abs(level), np.abs(e)))
    return out


def figure(name, label, e, fitted):
    nl = min(24, len(e) // 3)
    fig, ax = plt.subplots(3, 3, figsize=(11, 7.5))
    b = 1.96 / np.sqrt(len(e))
    for a, s, t in ((ax[0, 0], e, "ACF residuals"), (ax[0, 1], e ** 2, "ACF squared residuals")):
        a.bar(range(1, nl + 1), am.acf(s, nl)[1:], color="#52514e"); a.set_title(t, fontsize=9)
        a.axhline(b, color="r", lw=.6, ls="--"); a.axhline(-b, color="r", lw=.6, ls="--")
    ax[0, 2].bar(range(1, nl + 1), am.pacf(e, nl)[1:], color="#52514e"); ax[0, 2].set_title("PACF residuals", fontsize=9)
    ax[1, 0].bar(range(1, nl + 1), am.pacf(e ** 2, nl)[1:], color="#52514e"); ax[1, 0].set_title("PACF squared residuals", fontsize=9)
    stats.probplot(e, plot=ax[1, 1]); ax[1, 1].set_title("QQ plot", fontsize=9); ax[1, 1].set_xlabel(""); ax[1, 1].set_ylabel("")
    ax[1, 2].hist(e, bins=25, color="#1f4e9c"); ax[1, 2].set_title("histogram", fontsize=9)
    ax[2, 0].scatter(fitted, e, s=6, color="#d95f02"); ax[2, 0].set_title("residual vs fitted", fontsize=9)
    ax[2, 1].plot(e, lw=.8, color="#1f4e9c"); ax[2, 1].set_title("residual vs time", fontsize=9)
    ax[2, 2].scatter(np.abs(fitted), np.abs(e), s=6, color="#0f8a5f"); ax[2, 2].set_title("|residual| vs |level|", fontsize=9)
    fig.suptitle(f"{name}: {label}", fontsize=11)
    fig.tight_layout()
    fig.savefig(figure_path("residuals", f"{name}_{label}".replace(" ", "_").lower() + ".png"), dpi=100)
    plt.close(fig)


def analyse(name):
    warnings.filterwarnings("ignore")
    d = DATASETS[name]
    am.DEFAULT_SUBSET = d["arima_subset"]
    y, n = d["load"](), d["n_train"]
    arima = am.AutoARIMA(verbose=False).fit(y[:n])
    L = arima.predict_one_step(y)
    m0 = arima.max_lag
    sets = [("ARIMA train", (y - L)[m0:n], L[m0:n]), ("ARIMA test", (y - L)[n:], L[n:])]
    f = build("Zhang", d, registry_params("Zhang", d), 0).fit(y[:n]).predict_one_step(y)[n:]
    sets.append(("Hybrid test", y[n:] - f, f))
    rows = []
    for label, e, fitted in sets:
        rows.append(dict(dataset=name, residuals=label, n=len(e), **tests(e, fitted)))
        figure(name, label, e, fitted)
    return rows


def findings(df, joint):
    lines = ["# Residual findings", "", "Generated from `results/3_residuals_and_decomposition/3_residual_tests.csv` by `residual_diagnostics.py`; "
             f"tests at the {ALPHA:.0%} level.", ""]
    for _, j in joint.iterrows():
        name = j["dataset"]
        tr = df[(df.dataset == name) & (df.residuals == "ARIMA train")].iloc[0]
        lb = tr["ljung_box_p_12"] if pd.notna(tr.get("ljung_box_p_12")) else tr["ljung_box_p_6"]
        s = [f"ARIMA training residuals {'show no significant' if lb > ALPHA else 'show significant'} autocorrelation (Ljung-Box p = {lb:.3f})."]
        s.append(f"Nonlinearity tests on them give BDS p = {tr['bds_p']:.3f} and neural-network test p = {tr['nn_linearity_p']:.3f}, "
                 f"so nonlinearity is {'detected' if j['nonlinearity_detected'] else 'not detected'}.")
        s.append(f"ARCH-LM p = {tr['arch_lm_p']:.3f} ({'evidence of' if tr['arch_lm_p'] < ALPHA else 'no evidence of'} changing variance) and "
                 f"Jarque-Bera p = {tr['jarque_bera_p']:.3f} ({'not normal' if tr['jarque_bera_p'] < ALPHA else 'normality not rejected'}).")
        sp, sp_p = tr["var_vs_level_spearman"], tr["var_vs_level_p"]
        s.append(f"The size of the residuals {'rises significantly with' if sp > 0 and sp_p < ALPHA else 'does not rise significantly with'} the level "
                 f"(Spearman {sp:.2f}, p = {sp_p:.3f}).")
        s.append(f"The hybrid {'lowered' if j['hybrid_improved'] else 'did not lower'} the 10-seed mean test MSE "
                 f"({j['mse_arima']:.4g} for ARIMA, {j['mse_zhang']:.4g} for the hybrid).")
        lines += [f"## {name}", "", " ".join(s), ""]
    with open(os.path.join(ROOT, "RESIDUAL_FINDINGS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(mode="full"):
    names = MODES[mode]["datasets"] or list(DATASETS)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        rows = [r for rs in pool.map(analyse, names) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(results_path("residual_diagnostics.csv"), index=False)
    v = pd.read_csv(results_path("validation_results.csv"))
    v = v[(v.protocol == "fixed") & (v.window == "all test points") & (v.status == "ok")].groupby(["dataset", "model"])["MSE"].mean()
    joint = []
    for name in names:
        tr = df[(df.dataset == name) & (df.residuals == "ARIMA train")].iloc[0]
        det = bool(tr["bds_p"] < ALPHA or tr["nn_linearity_p"] < ALPHA)
        a, z = v[(name, "ARIMA")], v[(name, "Zhang")]
        joint.append(dict(dataset=name, nonlinearity_detected=det, hybrid_improved=bool(z < a), both=det == bool(z < a), mse_arima=a, mse_zhang=z))
    joint = pd.DataFrame(joint)
    joint.to_csv(results_path("nonlinearity_vs_improvement.csv"), index=False)
    findings(df, joint)
    print(joint.round(5).to_string(index=False))


if __name__ == "__main__":
    main("quick" if "--quick" in sys.argv else "full")
