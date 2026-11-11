# 1. Stationarity figures

**Files:** `<dataset>.png`, one per dataset, from the training part only. Written by `stationarity.py`.
**Numbers:** `results/1_stationarity/stationarity_tests.csv`.

Each figure has six panels, with the original series on the left and its first difference on the right:

| Row | Left: series | Right: first difference |
|---|---|---|
| 1 | Training series over time | Change from one step to the next |
| 2 | ACF: correlation of the series with its own past at each lag | ACF of the differenced series |
| 3 | PACF: correlation at each lag after removing the effect of shorter lags | PACF of the differenced series |

**How to read it:**
- If the series' ACF decays slowly while the differenced series' ACF drops inside the dashed red lines after a few
  lags, the series needs one difference (d = 1).
- Repeated spikes at the seasonal lag (12 for monthly data, 11 years for sunspots) show seasonality or cycles.
- The PACF lags that stand outside the red lines suggest the AR lags ARIMA will use.

**Why it is here:** ARIMA needs a stationary series after differencing. These figures show the reason behind the
number of differences `d` in the results table.
