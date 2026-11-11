# 2. Forecast decomposition figures

**Files:** three per dataset, from the test part (fixed protocol, seed 0). Written by `plots_decomposition.py`.

| File | Model |
|---|---|
| `<dataset>_zhang.png` | Zhang's hybrid |
| `<dataset>_extension.png` | The Extension |
| `<dataset>_best_variant.png` | The best-performing variant on that dataset: the one of E1–E7 with the lowest validation error, chosen without test data. The title names which variant it is, for example "best-performing variant (E3, …)". |

**Numbers:** `results/3_residuals_and_decomposition/2_correction_fit.csv`.

## Panels, from top to bottom

1. **Actual values, ARIMA forecast and final forecast.** Where the final forecast is closer to the actual line than
   the ARIMA forecast, the correction helped.
2. **ARIMA error and neural-network correction over time.** A useful correction follows the ARIMA error. A flat
   correction near zero means the network adds almost nothing.
3. **Neural-network correction (y-axis) against ARIMA error (x-axis), with R² in the title.** Points on the diagonal
   line mean the correction cancels the ARIMA error exactly.
   - R² near 1: the correction explains most of the ARIMA error.
   - R² near 0: it explains nothing.
   - R² below 0: it fits worse than adding no correction.
4. **Error left after the correction** (actual minus final forecast). Ideally smaller and less patterned than the
   ARIMA error.
5. **Change in squared error at each step due to the correction.** Green bars below zero are steps where the
   correction helped; orange bars above zero are steps where it hurt.

**Why it is here:** it tests Zhang's central assumption directly: can the network predict what ARIMA gets wrong?

**Example:** in `sunspot_zhang.png` the correction stays near zero (R² ≈ 0). In `synthetic_additive_extension.png`
it follows the ARIMA error closely (R² ≈ 0.90), which is where the Extension's large gain comes from.
