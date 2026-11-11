"""
Tests whether the Extension benefits from the ARIMA forecast L_hat_t as an input.
Runs Extension 1 (m = 0) and Extension 2 (m = 0.2) WITHOUT L_hat_t (include_linear_pred=False), inputs are
then only past values and past ARIMA errors, over 10 seeds, and compares them with the current versions
(with L_hat_t) and the hybrid, read from results_extensions.csv.
Usage:  python compare_no_lhat.py   (writes results_no_lhat.csv; --report re-prints from it)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor
import numpy as np
from config import results_path
import pandas as pd

SEEDS = range(10)
VARIANTS = {"Ext. 1 without L_hat": 0.0, "Ext. 2 without L_hat": 0.2}
CURRENT = {"Hybrid": "Zhang hybrid", "Ext. 1 with L_hat": "Extension",
           "Ext. 2 with L_hat": "Extension + margin"}


def run_one(job):
    from arima_model import AutoARIMA
    from extension_model import ExtensionHybrid
    from run_extension_experiments import EXPERIMENTS
    name, variant, seed = job
    cfg = EXPERIMENTS[name]
    y, n = cfg["load"](), cfg["n_train"]
    ext = dict(cfg["ext_kwargs"], seed=seed)
    model = ExtensionHybrid(AutoARIMA(verbose=False), include_linear_pred=False,
                                scale_margin=VARIANTS[variant], **ext)
    f = model.fit(y[:n]).predict_one_step(y)[n:]
    a = y[n:]
    row = {"dataset": name, "model": variant, "seed": seed}
    for h_name, h in cfg["horizons"].items():
        row[f"{h_name} MSE"] = np.mean((a[:h] - f[:h]) ** 2)
        row[f"{h_name} MAD"] = np.mean(np.abs(a[:h] - f[:h]))
    return row


def report(new):
    from run_extension_experiments import EXPERIMENTS
    old = pd.read_csv(results_path("results_extensions.csv"))
    old = old[old.model.isin(CURRENT.values())].copy()
    old["model"] = old.model.map({v: k for k, v in CURRENT.items()})
    df = pd.concat([old, new], ignore_index=True)
    order = ["Hybrid", "Ext. 1 with L_hat", "Ext. 1 without L_hat", "Ext. 2 with L_hat", "Ext. 2 without L_hat"]
    wins = {m: [0, 0] for m in order[1:]}
    for name, cfg in EXPERIMENTS.items():
        cols = [f"{h} {k}" for h in cfg["horizons"] for k in ("MSE", "MAD")]
        mean = df[df.dataset == name].groupby("model")[cols].mean().loc[order]
        scale = {c: (1e5 if name == "Exchange rate" and c.endswith("MSE") else 1.0) for c in cols}
        print(f"\n===== {name}" + (" (MSE x 1e5)" if name == "Exchange rate" else "") + " =====")
        print(f"{'':22}" + "".join(f"{c:>22}" for c in cols))
        for m in order:
            print(f"{m:22}" + "".join(f"{mean.loc[m, c] * scale[c]:>22.5g}" for c in cols))
        for m in order[1:]:
            wins[m][0] += int((mean.loc[m] < mean.loc["Hybrid"]).sum())
        for k in ("1", "2"):
            wins[f"Ext. {k} without L_hat"][1] += int(
                (mean.loc[f"Ext. {k} without L_hat"] < mean.loc[f"Ext. {k} with L_hat"]).sum())
    print("\n===== Cases (of 12) with lower mean error =====")
    for m in order[1:]:
        extra = f"   vs the same extension with L_hat: {wins[m][1]}/12" if "without" in m else ""
        print(f"{m:22} vs hybrid: {wins[m][0]}/12{extra}")


def main():
    from run_extension_experiments import EXPERIMENTS
    jobs = [(name, v, s) for name in EXPERIMENTS for v in VARIANTS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=12) as pool:
        rows = list(pool.map(run_one, jobs, chunksize=2))
    new = pd.DataFrame(rows)
    new.to_csv(results_path("results_no_lhat.csv"), index=False)
    report(new)


if __name__ == "__main__":
    import sys
    if "--report" in sys.argv:
        report(pd.read_csv(results_path("results_no_lhat.csv")))
    else:
        main()
