"""
Fails (exit code 1) if any value the research site reads is missing, null or not finite in website/research_site/data/.
The checks mirror what each section of the page uses.   Usage:  python website/check_site_data.py
"""
import json
import math
import os
import sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research_site", "data")
MODELS = ["ARIMA", "ANN", "Zhang", "Extension", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]
METRICS = ["MSE", "RMSE", "MAD", "MAPE", "sMAPE", "MASE", "TheilU2", "DA"]
problems = []


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def need(cond, msg):
    if not cond:
        problems.append(msg)


def numbers(values, where, allow_leading_null=False):
    vals = list(values)
    if allow_leading_null:
        while vals and vals[0] is None:
            vals = vals[1:]
    bad = [v for v in vals if v is None or not isinstance(v, (int, float)) or not math.isfinite(v)]
    need(vals and not bad, f"{where}: {len(bad)} missing or non-finite values")


index = load("datasets_index.json")
need(len(index) == 11, "datasets_index.json should list 11 datasets")
metrics, dm, mcs = load("metrics.json"), load("dm.json"), load("validation_mcs.json")
diag, stat = load("residual_diagnostics.json"), load("stationarity.json")
findings, nl = load("residual_findings.json"), load("nonlinearity_vs_improvement.json")
designs, singles, lead, when = load("grid_designs.json"), load("grid_single_models.json"), load("grid_leaderboard.json"), load("grid_when.json")
decomp = load("decomposition_r2.json")

for entry in index:
    name = entry["name"]
    d = load(f"datasets/{entry['slug']}.json")
    numbers(d["series"], f"{name} series")
    numbers(d["actual_test"], f"{name} actual_test")
    numbers(d["arima_forecast"], f"{name} arima_forecast")
    numbers(d["arima_error"], f"{name} arima_error")
    for proto in ("fixed", "rolling"):
        for m in MODELS + (["Extension margin"] if proto == "fixed" else []):
            md = d["models"].get(proto, {}).get(m)
            need(md is not None, f"{name} {proto} {m}: no forecasts")
            if md:
                for field in ("forecast", "final_error", "nn_correction"):
                    numbers(md[field], f"{name} {proto} {m} {field}")
            row = [r for r in metrics if r["protocol"] == proto and r["dataset"] == name and r["model"] == m]
            need(len(row) == 1, f"metrics: {proto} {name} {m} missing")
            if row:
                numbers([row[0][f"{k}_mean"] for k in METRICS], f"metrics {proto} {name} {m}")
            if m != "Zhang":
                q = [r for r in dm if r["protocol"] == proto and r["dataset"] == name and r["model"] == m and r["versus"] == "Zhang"]
                need(len(q) == 1 and q[0]["p_value"] is not None, f"dm: {proto} {name} {m} vs Zhang missing")
            need(any(r["protocol"] == proto and r["dataset"] == name and r["model"] == m for r in mcs), f"mcs: {proto} {name} {m} missing")
    need(any(r["dataset"] == name and r["model"] == "Zhang" and r["versus"] == "ARIMA" and r["protocol"] == "fixed" for r in dm),
         f"dm: Zhang vs ARIMA missing for {name}")
    for label in ("ARIMA train", "ARIMA test", "Hybrid test"):
        need(any(r["dataset"] == name and r["residuals"] == label for r in diag), f"residual_diagnostics: {name} {label} missing")
    need(bool(findings.get(name)), f"residual_findings: {name} missing")
    need(any(r["dataset"] == name for r in nl), f"nonlinearity_vs_improvement: {name} missing")
    st = [r for r in stat if r["dataset"] == name]
    need(len(st) == 1 and all(st[0][k] is not None for k in ("adf_p", "kpss_p", "pp_p", "d")), f"stationarity: {name} incomplete")
    need(any(r["dataset"] == name and r["model"] == "Zhang" for r in decomp), f"decomposition_r2: {name} Zhang missing")
    g = [r for r in designs if r["dataset"] == name]
    need(len(g) == 105, f"grid_designs: {name} has {len(g)} designs, expected 105")
    numbers([r["MSE"] for r in g], f"grid_designs {name}")
    need(any(r["dataset"] == name and r["MSE"] is not None for r in singles), f"grid_single_models: {name} missing")
    need(sum(r["dataset"] == name for r in lead) == 10, f"grid_leaderboard: {name} should have 10 rows")
    need(any(r["dataset"] == name for r in when), f"grid_when: {name} missing")

be = load("break_even.json")
need(len(be["cells"]) == len(be["strengths"]) * len(be["noises"]), "break_even: incomplete grid")
for c in be["cells"]:
    numbers([c["mse_mean"][m] for m in be["models"]], f"break_even cell {c['strength']},{c['noise']}")
lab = load("form_lab.json")
need(len(lab["rows"]) == len(lab["strengths"]) * len(lab["forms"]), "form_lab: incomplete")
for r in lab["rows"]:
    numbers([r["mse_mean"][m] for m in lab["models"]] + [r["mse_std"][m] for m in lab["models"]], f"form_lab {r['form']} {r['strength']}")
ra = load("rolling_animation.json")["datasets"]
for name in ("Lynx", "Nile"):
    d = ra[name]
    need(len(d["fitted_through"]) == len(d["test_index"]), f"rolling_animation {name}: lengths differ")
    for m in ("ARIMA", "Zhang", "Extension"):
        numbers(d["forecasts"][m], f"rolling_animation {name} {m}")
        need(len(d["forecasts"][m]) == len(d["test_index"]), f"rolling_animation {name} {m}: wrong length")
sf = load("seed_fan.json")["datasets"]
for name, d in sf.items():
    for m in ("Zhang", "Extension"):
        need(len(d["forecasts"][m]) == 10, f"seed_fan {name} {m}: expected 10 seeds")
        for s, f in d["forecasts"][m].items():
            numbers(f, f"seed_fan {name} {m} seed {s}")
sm = load("scale_margin.json")
for m in sm["margins"]:
    f = sm["forecasts"][f"{m:g}"]
    numbers(f["forecast"] + [f["mse"], f["mse_record_points"], f["max_forecast"]], f"scale_margin {m}")
need(os.path.exists(os.path.join(os.path.dirname(DATA), "report.pdf")), "report.pdf missing from the site folder")

if problems:
    print(f"{len(problems)} problems:")
    print("\n".join("  " + p for p in problems))
    sys.exit(1)
print("every value the site needs is present and finite")
