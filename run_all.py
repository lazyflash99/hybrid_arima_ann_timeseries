"""
Regenerates every table and figure.
  python run_all.py --quick     3 data sets, 2 seeds, small grids (minutes)
  python run_all.py --full      all data sets, 10 seeds, full grids, plus the paper-replication scripts
Library versions and step times are written to results/run_log.json.
The website data (website/data/) is written by the last export step.
"""
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version

from config import ROOT, results_path

PACKAGES = ["numpy", "pandas", "scipy", "statsmodels", "matplotlib", "scikit-learn", "arch", "PyWavelets", "lightgbm", "torch"]


def versions():
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = "not installed"
    return out


def main():
    quick = "--full" not in sys.argv
    flag = ["--quick"] if quick else []
    write_md = [sys.executable, "-c", "import data_loader as d; d.write_datasets_md('DATASETS.md')"]
    steps = [("datasets", write_md), ("stationarity", [sys.executable, "stationarity.py"] + flag),
             ("validation", [sys.executable, "validation.py"] + flag),
             ("decomposition", [sys.executable, "plots_decomposition.py"] + flag),
             ("residuals", [sys.executable, "residual_diagnostics.py"] + flag),
             ("framework", [sys.executable, "hybrid_framework.py"] + flag),
             ("paper replication", [sys.executable, "run_experiments.py"])]
    if not quick:
        steps += [("extension experiments", [sys.executable, "run_extension_experiments.py"]),
                  ("all variants, 10 seeds", [sys.executable, "run_extensions.py"]),
                  ("extension report", [sys.executable, "make_extension_report.py"]),
                  ("margin tuning", [sys.executable, "tune_scale_margin.py"]),
                  ("error-input tuning", [sys.executable, "tune_error_inputs.py"]),
                  ("past-value tuning", [sys.executable, "tune_past_values.py"]),
                  ("input-design tests", [sys.executable, "compare_arima_lags.py"]),
                  ("no-forecast-input test", [sys.executable, "compare_no_lhat.py"]),
                  ("error figure", [sys.executable, "make_error_figure.py"]),
                  ("slide figures", [sys.executable, "make_slide_figures.py"])]
    steps += [("summary", [sys.executable, "make_results_summary.py"]), ("website data", [sys.executable, "website/export_json.py"]),
              ("result checks", [sys.executable, "check_results.py"])]
    if not quick:
        steps += [("report tables", [sys.executable, "report/make_report_tables.py"])]
    log = {"mode": "quick" if quick else "full", "versions": versions(), "steps": {}}
    print(json.dumps(log["versions"]))
    for name, cmd in steps:
        t = time.time()
        print(f"\n=== {name} ===", flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        log["steps"][name] = round(time.time() - t, 1)
        if r.returncode:
            print(f"step failed: {name}")
            sys.exit(r.returncode)
    with open(results_path("run_log.json"), "w") as f:
        json.dump(log, f, indent=1)
    print("\nall steps finished:", log["steps"])


if __name__ == "__main__":
    main()
