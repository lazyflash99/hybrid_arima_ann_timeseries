"""
Variant of the extensions whose inputs follow the fitted ARIMA model instead of the paper's ANN sizes:
past values at ARIMA's AR lags, ARIMA's forecast L_hat_t, and past ARIMA errors at the same lags.
  sunspot  (AR lags 1, 2, 9)  -> 3 values + L_hat + 3 errors = 7 inputs
  lynx     (AR lags 1, 2, 10) -> 3 values + L_hat + 3 errors = 7 inputs
  exchange (random walk, y_t = y_{t-1} + e_t, i.e. lag 1 in levels) -> 1 + 1 + 1 = 3 inputs
Hidden nodes, training (weight penalty, 20 restarts, L-BFGS-B) and margins are unchanged.
Compared with the current extensions (results_extensions.csv) over 10 seeds.
Usage:  python compare_arima_lags.py   (writes results_arima_lags.csv; --report re-prints from it)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor
import numpy as np
from config import results_path
import pandas as pd

from extension_model import ExtensionHybrid

SEEDS = range(10)
VARIANTS = {"Ext. 1, ARIMA lags": 0.0, "Ext. 2, ARIMA lags": 0.2}
CURRENT = {"Hybrid": "Zhang hybrid", "Ext. 1, paper sizes": "Extension",
           "Ext. 2, paper sizes": "Extension + margin"}


class ArimaLagHybrid(ExtensionHybrid):
    """Inputs at the fitted ARIMA's AR lags (lag 1 for a random walk) for both past values and past errors."""

    def lags(self):
        return list(self.linear.model.ar_lags) or [1]

    def _get_start_idx(self):
        k = max(self.lags())
        return max(k, self.linear.max_lag + k)          # every past value and past error must exist

    def _build_feature_matrix(self, y_series, L_hat, e_series, start_idx, end_idx=None):
        end_idx = len(y_series) if end_idx is None else end_idx
        lags = self.lags()
        indices = np.arange(start_idx, end_idx)
        X = np.array([[self._scale_y(y_series[t - k]) for k in lags] + [self._scale_y(L_hat[t])]
                      + [self._scale_e(e_series[t - k]) for k in lags] for t in indices], dtype=float)
        return X, self._scale_y(y_series[indices].astype(float)), indices


def run_one(job):
    from arima_model import AutoARIMA
    from run_extension_experiments import EXPERIMENTS
    name, variant, seed = job
    cfg = EXPERIMENTS[name]
    y, n = cfg["load"](), cfg["n_train"]
    ext = dict(cfg["ext_kwargs"], seed=seed)
    model = ArimaLagHybrid(AutoARIMA(verbose=False), scale_margin=VARIANTS[variant], **ext).fit(y[:n])
    f = model.predict_one_step(y)[n:]
    a = y[n:]
    row = {"dataset": name, "model": variant, "seed": seed, "n_inputs": 2 * len(model.lags()) + 1}
    for h_name, h in cfg["horizons"].items():
        row[f"{h_name} MSE"] = np.mean((a[:h] - f[:h]) ** 2)
        row[f"{h_name} MAD"] = np.mean(np.abs(a[:h] - f[:h]))
    return row


def report(new):
    from run_extension_experiments import EXPERIMENTS
    old = pd.read_csv(results_path("results_extensions.csv"))
    old = old[old.model.isin(CURRENT.values())].copy()
    old["model"] = old.model.map({v: k for k, v in CURRENT.items()})
    df = pd.concat([old, new.drop(columns="n_inputs")], ignore_index=True)
    order = ["Hybrid", "Ext. 1, paper sizes", "Ext. 1, ARIMA lags", "Ext. 2, paper sizes", "Ext. 2, ARIMA lags"]
    wins = {m: [0, 0] for m in order[1:]}
    for name, cfg in EXPERIMENTS.items():
        cols = [f"{h} {k}" for h in cfg["horizons"] for k in ("MSE", "MAD")]
        mean = df[df.dataset == name].groupby("model")[cols].mean().loc[order]
        scale = {c: (1e5 if name == "Exchange rate" and c.endswith("MSE") else 1.0) for c in cols}
        n_in = int(new[new.dataset == name].n_inputs.iloc[0])
        print(f"\n===== {name}" + (" (MSE x 1e5)" if name == "Exchange rate" else "")
              + f" -- ARIMA-lag variant has {n_in} inputs =====")
        print(f"{'':22}" + "".join(f"{c:>22}" for c in cols))
        for m in order:
            print(f"{m:22}" + "".join(f"{mean.loc[m, c] * scale[c]:>22.5g}" for c in cols))
        for m in order[1:]:
            wins[m][0] += int((mean.loc[m] < mean.loc["Hybrid"]).sum())
        for k in ("1", "2"):
            wins[f"Ext. {k}, ARIMA lags"][1] += int(
                (mean.loc[f"Ext. {k}, ARIMA lags"] < mean.loc[f"Ext. {k}, paper sizes"]).sum())
    print("\n===== Cases (of 12) with lower mean error =====")
    for m in order[1:]:
        extra = f"   vs paper-size version: {wins[m][1]}/12" if "ARIMA" in m else ""
        print(f"{m:22} vs hybrid: {wins[m][0]}/12{extra}")


def main():
    from run_extension_experiments import EXPERIMENTS
    jobs = [(name, v, s) for name in EXPERIMENTS for v in VARIANTS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(run_one, jobs, chunksize=2))
    new = pd.DataFrame(rows)
    new.to_csv(results_path("results_arima_lags.csv"), index=False)
    report(new)


if __name__ == "__main__":
    import sys
    if "--report" in sys.argv:
        report(pd.read_csv(results_path("results_arima_lags.csv")))
    else:
        main()
