# 5. Original three datasets

**Question:** how do the Extension and the variants compare with the paper's models on the paper's own datasets,
and which input sizes and scale margin should the Extension use?
**Datasets:** Sunspot, Lynx and Exchange rate only, with the fixed protocol (one fit on the training part,
one-step forecasts on the test part).
**Figures:** `figures/5_original_three_datasets/`.

These files are in **long format**. Each dataset has its own error windows (Sunspot: 35 and 67 points ahead; Lynx:
14 points ahead; Exchange rate: 1, 6 and 12 months), so the columns for another dataset's windows are empty. A
window is the average of one-step forecasts over the first `k` test points (see
`../2_model_comparison/README.md`). The column `<window> MSE` holds the mean squared error and `<window> MAD` the
mean absolute error.

In files 3–7 the model names "Extension 1" and "Extension 2" (or "Ext. 1", "Ext. 2") are the names used on the
slides for the Extension and the Extension + margin. "Hybrid" is Zhang's hybrid.

## Files

| File | Written by | Content |
|---|---|---|
| `1_single_run_metrics_seed0.csv` | `make_extension_report.py` | **Single run with seed 0:** Zhang's hybrid, the Extension and the Extension + margin, each trained once from random starting point 0, as in the paper's tables. Gives the errors per window plus extra measures and a significance test; see below. |
| `2_all_models_10_seeds.csv` | `run_extensions.py` | ARIMA, ANN, Zhang, every Extension option and E1–E7, each network run with seeds 0–9: one row per dataset, model and seed. |
| `3_scale_margin_tuning.csv` | `tune_scale_margin.py` | Scale margins `m` from 0 to 0.3, seeds 0–9. |
| `4_past_error_inputs_tuning.csv` | `tune_error_inputs.py` | Number of past ARIMA errors `k` given to the networks (1, 2, 3, 4, 7). |
| `5_past_value_inputs_tuning.csv` | `tune_past_values.py` | Number of past values `k` given to the networks. |
| `6_without_arima_forecast_input.csv` | `compare_no_lhat.py` | The Extension without the ARIMA forecast as an input, to test whether that input matters. |
| `7_inputs_at_arima_lags.csv` | `compare_arima_lags.py` | The Extension with inputs taken at ARIMA's own lags instead of the paper's sizes. |

In files 3–5 the column `stage` shows how each row was used:
- `validation` rows: the model is trained on the first 80% of the training part and scored on the last 20%. The
  setting with the lowest mean validation MSE is chosen.
- `test` rows: test errors for every setting, kept only to show how sensitive the result is. They were never used
  to choose.

## `1_single_run_metrics_seed0.csv` in detail

"Single run, seed 0" means each network is trained once, starting from the random weights given by seed 0, as the
paper reports one run per model. Results from one run depend on that starting point; file 2 shows the spread over
seeds 0–9.

| Column | Meaning |
|---|---|
| `<window> MSE`, `<window> MAD` | Errors over each of the paper's windows. |
| `RMSE` | Square root of the MSE over the whole test part, in the units of the series. |
| `MAPE (%)`, `sMAPE (%)` | Percentage errors (`100·mean(|e|/|a|)` and `100·mean(2|e|/(|a|+|f|))`). |
| `MASE` | Mean absolute error ÷ mean absolute change of the training series; below 1 beats the naive "last value" forecast. |
| `Theil U2` | `√(MSE / MSE of the random walk)` on the test part; below 1 beats "no change". |
| `DA (%)` | Share of steps where the predicted and actual changes have the same sign. |
| `DM vs Zhang`, `DM p-value` | Diebold–Mariano test against Zhang's hybrid; see below. Empty in Zhang's own row. |

**Diebold–Mariano test:** it checks whether two forecasts of the same test points differ in accuracy by more than
chance.
- With `d_t` = Zhang's squared error minus this model's squared error at test point `t`, the statistic is
  `mean(d) / √(var(d)/T)` with the Harvey–Leybourne–Newbold small-sample correction.
- It is compared with a t distribution with `T−1` degrees of freedom.
- **A positive `DM vs Zhang` means this model has the lower error. A `DM p-value` below 0.05 means the difference
  is significant;** above 0.05 the test data cannot tell the two apart.
