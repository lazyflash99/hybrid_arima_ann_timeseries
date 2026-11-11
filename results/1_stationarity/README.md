# 1. Stationarity

**Question:** is each series stationary, and how many differences does ARIMA need?
**Written by:** `stationarity.py`, using the training part of each dataset only.
**Figures:** `figures/1_stationarity/`.

ARIMA assumes the series is stationary after differencing (constant mean and variance over time). Too little
differencing leaves a trend in the errors; too much adds noise. This table records the evidence behind the `d`
that ARIMA uses.

## `stationarity_tests.csv` (one row per dataset)

| Column | Meaning |
|---|---|
| `n_train` | Length of the training part. |
| `adf_p` | Augmented Dickey–Fuller test. The null hypothesis is a unit root (non-stationary). **Below 0.05: stationary.** |
| `kpss_p` | KPSS test. The null hypothesis is stationarity. **Below 0.05: not stationary.** Values are bounded to 0.01–0.10 by the test's lookup table. |
| `pp_p` | Phillips–Perron test; same null hypothesis and reading as ADF. |
| `adf_kpss` | Whether ADF and KPSS give the same verdict (`agree: stationary`, `agree: non-stationary`, or `disagree`). |
| `decision_note` | Filled only when ADF and KPSS disagree: it states that the pipeline follows ADF. |
| `rolling_mean_spread` | Standard deviation of the rolling mean divided by the standard deviation of the series. Large values: the level drifts. |
| `rolling_var_ratio` | Largest rolling variance divided by the smallest. Large values: the variance changes over time. |
| `seasonal_strength` | 0 (no seasonality) to 1 (strong); see below. Empty when the dataset has no seasonal period. |
| `boxcox_lambda_shifted` | Box–Cox λ of the series shifted to positive values. Near 1: no transformation needed; near 0: a log transformation. |
| `d` | Number of ordinary differences: the smallest number for which ADF rejects a unit root. Used by ARIMA. |
| `D` | Number of seasonal differences (1 when seasonal strength ≥ 0.64). Used only by SARIMA in the grid. |
| `ar_lags`, `ma_lags` | Lags of the ARIMA model selected on the training data. |
| `arima_passes_ljung_box` | Whether that model's errors are free of autocorrelation (Ljung–Box at lags 6, 12, 18 and 24), i.e. whether the model is adequate. |

**Rolling window:** the seasonal period when it is between 2 and a third of the series; otherwise a tenth of the
series, and at least 8 points.

**Seasonal strength:** the training series is split with an STL decomposition into trend, seasonal part `S` and
remainder `R`, and

    seasonal strength = max(0, 1 − Var(R) / Var(S + R))      (Hyndman's F_s)

It is close to 1 when the seasonal pattern dominates the remainder. It is empty when the seasonal period is 1 or
the series is shorter than two full periods.
