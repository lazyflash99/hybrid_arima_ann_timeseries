# 3. Residual diagnostics figures

**Files:** three per dataset. Written by `residual_diagnostics.py`.

| File | Errors shown |
|---|---|
| `<dataset>_arima_train.png` | ARIMA's errors on the training part |
| `<dataset>_arima_test.png` | ARIMA's errors on the test part |
| `<dataset>_hybrid_test.png` | Zhang's hybrid's errors on the test part |

**Test p-values:** `results/3_residuals_and_decomposition/3_residual_tests.csv`. **Short text per dataset:**
`RESIDUAL_FINDINGS.md` in the project root.

## Panels (3 × 3)

| Panel | What to look for |
|---|---|
| ACF of errors | Bars outside the dashed red lines: autocorrelation is left, so the model missed linear structure. |
| ACF of squared errors | Bars outside the lines: large errors cluster in time (changing variance, ARCH effects). |
| PACF of errors | Which lag the leftover autocorrelation comes from. |
| PACF of squared errors | Which lag the clustering of large errors comes from. |
| QQ plot | Points on the line: normally distributed errors. Points bending away at the ends: heavy tails, i.e. more large errors than a normal distribution has. |
| Histogram | Shape of the error distribution: centre, skew, outliers. |
| Error against fitted value | A funnel or curve means the error depends on the forecast level. |
| Error against time | Calm and volatile periods, or a drift the model missed. |
| Absolute error against absolute level | An upward trend means errors grow with the level of the series, a sign of a multiplicative structure. |

**Why it is here:** a hybrid can only help if ARIMA's errors contain a learnable pattern. These plots show what
kind of pattern is left, if any, and whether the hybrid removes it.
