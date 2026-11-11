"""
Copies exactly the JSON the research site needs from website/data/ into website/research_site/data/, filtering the
large tables to the rows the site uses and averaging the grid over seeds. No number is computed by hand here: every
value comes from website/data/ (written by export_json.py and website_experiments.py) or from RESIDUAL_FINDINGS.md.

Usage:  python website/build_site_data.py      (after export_json.py and website_experiments.py)
"""
import json
import os
import re
import shutil

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(HERE, "data")
DST = os.path.join(HERE, "research_site", "data")
DATASETS = ["Sunspot", "Lynx", "Exchange rate", "AirPassengers", "Nile", "Daily min temp", "FX JPY", "Sunspot sqrt",
            "Synthetic additive", "Synthetic multiplicative", "Mackey-Glass"]


def slug(name):
    return name.replace(" ", "_").lower()


def load(name):
    with open(os.path.join(SRC, name), encoding="utf-8") as f:
        return json.load(f)


def save(name, obj):
    path = os.path.join(DST, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, allow_nan=False, separators=(",", ":"))


def records(df):
    return json.loads(df.to_json(orient="records", double_precision=6))


def findings():
    """One paragraph per dataset from RESIDUAL_FINDINGS.md."""
    text = open(os.path.join(ROOT, "RESIDUAL_FINDINGS.md"), encoding="utf-8").read()
    out = {}
    for block in re.split(r"\n## ", text)[1:]:
        title, body = block.split("\n", 1)
        # the markdown prints p-values below 0.0005 as "p = 0.000"; the site writes them as "p < 0.001"
        out[title.strip()] = re.sub(r"p = 0\.000\b", "p < 0.001", " ".join(body.split()))
    return out


def main():
    if os.path.isdir(DST):
        shutil.rmtree(DST)
    # per-dataset files, loaded by the site one at a time
    index = []
    for name in DATASETS:
        d = load(slug(name) + ".json")
        save(f"datasets/{slug(name)}.json", d)
        index.append(dict(name=name, slug=slug(name), source=d["source"], frequency=d["frequency"], transform=d["transform"],
                          n_train=d["n_train"], n_test=len(d["actual_test"]), n_total=len(d["series"])))
    save("datasets_index.json", index)

    summary = pd.DataFrame(load("validation_summary.json"))
    keep = ["protocol", "dataset", "model"] + [c for c in summary.columns if c.endswith("_mean") or c.endswith("_std")]
    save("metrics.json", records(summary[summary.window == "all test points"][keep]))
    dm = pd.DataFrame(load("validation_dm.json"))
    save("dm.json", records(dm[["protocol", "dataset", "model", "versus", "dm_stat", "p_value"]]))

    for name in ("stationarity", "residual_diagnostics", "nonlinearity_vs_improvement", "grid_leaderboard", "grid_when",
                 "grid_single_models", "decomposition_r2", "validation_mcs"):
        save(f"{name}.json", load(f"{name}.json"))

    grid = pd.DataFrame(load("grid_results.json"))
    grid = grid[grid.status == "ok"].groupby(["dataset", "linear", "nonlinear", "combiner"], as_index=False).agg(
        MSE=("MSE", "mean"), MSE_std=("MSE", "std"), n_runs=("MSE", "size"))
    grid["MSE_std"] = grid["MSE_std"].fillna(0.0)          # one run (SVR is deterministic): spread is zero
    save("grid_designs.json", records(grid))

    save("residual_findings.json", findings())
    for name in ("break_even", "form_lab", "rolling_animation", "seed_fan", "scale_margin"):
        save(f"{name}.json", load(f"{name}.json"))

    shutil.copy(os.path.join(ROOT, "report", "main.pdf"), os.path.join(HERE, "research_site", "report.pdf"))
    n = sum(len(fs) for _, _, fs in os.walk(DST))
    print(f"wrote {n} JSON files to {DST} and copied report.pdf")


if __name__ == "__main__":
    main()
