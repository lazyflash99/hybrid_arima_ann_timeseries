"""
Stationarity analysis of every data set (training part only, so nothing about the test period is used).
ADF, KPSS, Phillips-Perron, rolling mean/variance, seasonal strength, Box-Cox lambda, and the d (and D) chosen.
Usage:  python stationarity.py [--quick]    writes results/1_stationarity/ and figures/1_stationarity/
"""
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from arch.unitroot import PhillipsPerron

import arima_model as am
from config import MODES, results_path, figure_path
from data_loader import DATASETS

ALPHA = 0.05


def seasonal_strength(y, period):
    """Hyndman's F_s = max(0, 1 - Var(R) / Var(S + R)); NaN when the data set has no season or is too short."""
    if period <= 1 or len(y) < 2 * period + 1:
        return np.nan
    fit = STL(y, period=period, robust=True).fit()
    return float(max(0.0, 1.0 - np.var(fit.resid) / np.var(fit.seasonal + fit.resid)))


def rolling_check(y, window):
    """Ratio of the spread of the rolling mean (to the series s.d.) and max / min of the rolling variance."""
    s = pd.Series(y)
    mean, var = s.rolling(window).mean().dropna(), s.rolling(window).var().dropna()
    return float(mean.std() / s.std()), float(var.max() / max(var.min(), 1e-12))


def decide(adf_p, kpss_p):
    """Which reading the tests give; the pipeline itself always takes d from the ADF test (arima_model.choose_d)."""
    adf_st, kpss_st = adf_p < ALPHA, kpss_p > ALPHA
    if adf_st and kpss_st:
        return "agree: stationary", ""
    if not adf_st and not kpss_st:
        return "agree: non-stationary", ""
    return "disagree", ("ADF says stationary, KPSS says not" if adf_st else "ADF says unit root, KPSS says stationary") \
        + "; pipeline follows ADF (d from choose_d)"


def analyse(name, d):
    y = d["load"]()[:d["n_train"]]
    period = d["period"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adf_p = adfuller(y, regression="c", autolag="AIC")[1]
        kpss_p = kpss(y, regression="c", nlags="auto")[1]
        pp_p = float(PhillipsPerron(y).pvalue)
    agreement, note = decide(adf_p, kpss_p)
    fs = seasonal_strength(y, period)
    window = max(8, period if 1 < period < len(y) / 3 else len(y) // 10)
    mean_spread, var_ratio = rolling_check(y, window)
    lam = float(boxcox(y - y.min() + 1.0)[1])
    am.DEFAULT_SUBSET = d["arima_subset"]
    m = am.AutoARIMA(verbose=False).fit(y).model
    capital_d = int(fs >= 0.64) if np.isfinite(fs) else 0
    return dict(dataset=name, n_train=len(y), adf_p=adf_p, kpss_p=kpss_p, pp_p=pp_p, adf_kpss=agreement, decision_note=note,
                rolling_mean_spread=mean_spread, rolling_var_ratio=var_ratio, seasonal_strength=fs,
                boxcox_lambda_shifted=lam, d=m.d, D=capital_d, ar_lags=str(m.ar_lags), ma_lags=str(m.ma_lags),
                arima_passes_ljung_box=bool(m.diagnostics and all(r[3] > ALPHA for r in m.diagnostics)))


def plot(name, d, row):
    y = d["load"]()[:d["n_train"]]
    w = np.diff(y)
    nl = min(40, len(y) // 3)
    fig, ax = plt.subplots(3, 2, figsize=(9, 6))
    ax[0, 0].plot(y, lw=.9, color="#1f4e9c"); ax[0, 0].set_title(f"{name}: training series", fontsize=9)
    ax[0, 1].plot(w, lw=.8, color="#d95f02"); ax[0, 1].set_title("first difference", fontsize=9)
    for col, s in enumerate((y, w)):
        ax[1, col].bar(range(1, nl + 1), am.acf(s, nl)[1:], color="#52514e"); ax[1, col].set_title("ACF", fontsize=9)
        ax[2, col].bar(range(1, nl + 1), am.pacf(s, nl)[1:], color="#52514e"); ax[2, col].set_title("PACF", fontsize=9)
        for r in (1, 2):
            b = 1.96 / np.sqrt(len(s))
            ax[r, col].axhline(b, color="r", lw=.6, ls="--"); ax[r, col].axhline(-b, color="r", lw=.6, ls="--")
    fig.tight_layout()
    fig.savefig(figure_path("stationarity", name.replace(" ", "_").lower() + ".png"), dpi=110)
    plt.close(fig)


def main(mode="full"):
    names = MODES[mode]["datasets"] or list(DATASETS)
    rows = []
    for name in names:
        row = analyse(name, DATASETS[name]); plot(name, DATASETS[name], row); rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(results_path("stationarity.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df[["dataset", "adf_p", "kpss_p", "pp_p", "adf_kpss", "seasonal_strength", "d", "D", "ar_lags", "ma_lags",
                  "arima_passes_ljung_box"]].round(3).to_string(index=False))
    return df


if __name__ == "__main__":
    main("quick" if "--quick" in sys.argv else "full")
