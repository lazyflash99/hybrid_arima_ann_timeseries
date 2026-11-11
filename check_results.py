"""
Scans every CSV in results/ for missing and infinite values and states why each missing value is expected.
Exits with an error on any infinite value or any missing value without a known reason.   Usage:  python check_results.py
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

from config import ROOT

SEASON = "the data set has no seasonal period"
RULES = [
    ("5_errors_mean_std_per_window.csv", r".*_std$", "standard deviation over one run: ARIMA is deterministic"),
    ("1_test_mse_overview.csv", r"Extension margin", "Extension margin is run in the fixed protocol only"),
    ("1_best_hybrid_per_dataset.csv", r"seasonal_strength", SEASON),
    ("stationarity_tests.csv", r"seasonal_strength", SEASON),
    ("stationarity_tests.csv", r"decision_note", "the note is written only when ADF and KPSS disagree"),
    ("3_residual_tests.csv", r"(ljung_box|box_pierce)_p_12|bds_p", "test series too short for this statistic"),
    ("3_residual_tests.csv", r"var_vs_level_(spearman|p)", "the ARIMA forecast is constant (no AR or MA terms), so the rank correlation is undefined"),
    (r"[1-7]_(single_run|all_models|scale_margin|past_error|past_value|without|inputs_at).*", r".* (MSE|MAD)$", "long format: each dataset has only its own horizon columns"),
    (r"[3-5]_(scale_margin|past_error|past_value).*", r"MSE|MAD", "validation rows have MSE and MAD; test rows have horizon columns"),
    ("1_single_run_metrics_seed0.csv", r"DM.*", "no Diebold-Mariano comparison for this row (the row is the reference or ARIMA)"),
]


def reason(fname, col):
    for pattern, cols, why in RULES:
        if re.fullmatch(pattern, fname) and re.fullmatch(cols, col):
            return why
    return None


def main():
    problems = 0
    files = sorted(glob.glob(os.path.join(ROOT, "results", "**", "*.csv"), recursive=True))
    for path in files:
        fname, df = os.path.basename(path), pd.read_csv(path)
        if "status" in df:                                   # a run that did not complete has no metrics; its status says why
            not_run = df[df.status != "ok"]
            for s, n in not_run.status.value_counts().items():
                print(f"{fname}: {n} rows not run: {s}")
            df = df[df.status == "ok"]
        inf = np.isinf(df.select_dtypes("number")).sum()
        for col, n in inf[inf > 0].items():
            print(f"PROBLEM {fname}: {n} infinite values in {col}")
            problems += 1
        na = df.isna().sum()
        for col, n in na[na > 0].items():
            why = reason(fname, col)
            print(f"{fname}: {n} missing in {col}: {why}" if why else f"PROBLEM {fname}: {n} missing in {col}, no known reason")
            problems += why is None
    print(f"checked {len(files)} files, {problems} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
