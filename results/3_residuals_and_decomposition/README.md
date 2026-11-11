# 3. Residuals and decomposition

**Question:** do ARIMA's errors still contain a pattern, and how much of it does the neural network correct?
**Written by:**
- `residual_diagnostics.py`: files 1 and 3, plus `RESIDUAL_FINDINGS.md` in the project root, which has a short
  text per dataset.
- `plots_decomposition.py`: file 2.

**Figures:** `figures/2_forecast_decomposition/` and `figures/3_residual_diagnostics/`.

All files use the fixed protocol. File 2 uses seed 0.

## Terms

- **ARIMA error:** actual value minus ARIMA forecast. Zhang's hybrid assumes this error still holds a pattern that
  a network can learn.
- **Neural-network correction:** what a hybrid adds to the ARIMA forecast, i.e. its final forecast minus the ARIMA
  forecast.

## Files

### `1_nonlinearity_vs_hybrid_gain.csv` (11 rows, start here)

Does detecting nonlinearity in ARIMA's errors predict whether Zhang's hybrid helps?

| Column | Meaning |
|---|---|
| `nonlinearity_detected` | True when the BDS test or the neural-network test on ARIMA's training errors has p < 0.05. |
| `hybrid_improved` | True when Zhang's hybrid has a lower mean test MSE than ARIMA. |
| `both` | True when the two columns above give the same answer: both yes or both no. False means the test and the outcome disagree. |
| `mse_arima`, `mse_zhang` | The two test MSEs being compared. |

### `2_correction_fit.csv` (33 rows)

For Zhang's hybrid, the Extension and the best-performing variant on each dataset. The best-performing variant is
the one of E1–E7 with the lowest validation error, chosen without test data.

| Column | Meaning |
|---|---|
| `model` | Zhang, Extension, or the name of the best-performing variant. |
| `r2_correction_vs_arima_error` | How much of the ARIMA error the correction explains: 1 = all of it, 0 = nothing, negative = worse than adding no correction. |
| `mse_arima`, `mse_model` | Test MSE of ARIMA alone and of the model. |
| `share_steps_improved` | Share of test points where the correction reduced the squared error. |

### `3_residual_tests.csv` (33 rows)

Three rows per dataset, given by `residuals`:
- `ARIMA train`: ARIMA's errors on the training part.
- `ARIMA test`: ARIMA's errors on the test part.
- `Hybrid test`: Zhang's hybrid's errors on the test part.

`n` is the number of errors tested. All other columns are p-values, except the two Spearman columns.

| Test (columns) | Question | p below 0.05 means |
|---|---|---|
| Ljung–Box, Box–Pierce at lags 6 and 12 (`ljung_box_p_6`, …) | Are the errors autocorrelated? | Pattern left over: the model missed linear structure. |
| Jarque–Bera, Shapiro–Wilk | Are the errors normally distributed? | Not normal (skewed or heavy-tailed). |
| ARCH-LM (`arch_lm_p`) | Does the error variance change over time? | Calm and volatile periods alternate. |
| BDS (`bds_p`) | Are the errors independent? | Some dependence remains, possibly nonlinear. |
| Neural-network test of Teräsvirta et al. (`nn_linearity_p`) | Is there a nonlinear pattern a network could learn? | Nonlinearity detected. |
| `var_vs_level_spearman`, `var_vs_level_p` | Do errors grow with the level of the series? (Spearman correlation of absolute error with the fitted level) | A positive, significant correlation suggests a multiplicative structure. |

Some cells are empty for a stated reason:
- The test part is too short for lag 12 or for the BDS test.
- The ARIMA forecast is constant (Synthetic multiplicative), so the correlation with the level is undefined.

`python check_results.py` lists each empty cell.
