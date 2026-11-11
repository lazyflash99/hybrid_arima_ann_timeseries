# 2. Model comparison

**Question:** which model forecasts best on the test part, and is the difference significant?
**Written by:** `validation.py`.
**Models:** ARIMA, ANN, Zhang, Extension, Extension margin, and E1–E7 on all 11 datasets.

## The two protocols

Every model is evaluated twice. The `protocol` column says which evaluation a row belongs to.

| Protocol | How the model is fitted | Why |
|---|---|---|
| `fixed` | Zhang's protocol. One fit on the whole training part, with network sizes fixed in advance (the paper's sizes for its three datasets; for the other eight, sizes set before any test result was seen). One-step forecasts over the whole test part. | Reproduces the paper's setup. |
| `rolling` | Stricter. The settings (hidden units, lags, and for the Extension also the scale margin and weight decay) are chosen on 2 validation folds inside the last 20% of the training part. The test part is then forecast with an expanding window, and the model is refitted 3 times as it moves through the test part. | No decision uses test data, so the result shows how the method would do in practice. |

In both protocols every network is run with seeds 0–9. The test part is never used to choose a setting, a scale
or a weight. "Extension margin" (margin fixed at 0.2) exists only in the fixed protocol, because the rolling
protocol tunes the margin.

## Files

| File | Rows | Content |
|---|---|---|
| `1_test_mse_overview.csv` | 22 | **Start here.** Mean test MSE over seeds and all test points: one row per protocol and dataset, one column per model. |
| `2_chosen_hyperparameters.csv` | 121 | Rolling protocol: the settings chosen for each dataset and model (`params`) and their validation MSE (`val_mse`). |
| `3_significance_diebold_mariano.csv` | 462 | Diebold–Mariano test of every model against Zhang and against ARIMA (`versus`), with both MSEs (`model_mse`, `ref_mse`). |
| `4_model_confidence_set.csv` | 253 | Model confidence set p-value (`mcs_p`) and whether the model is in the 90% set (`in_90pct_set`). |
| `5_errors_mean_std_per_window.csv` | 1,024 | All eight accuracy measures as mean and standard deviation over seeds, for every window. The standard deviation is empty for ARIMA. |
| `6_raw_every_run.csv` | 9,430 | One row per protocol, dataset, model, seed and window. This is the source of files 1 and 5. `status` is `ok` for every row, and `seconds` is the run time. |
| `7_raw_forecasts_fixed_split.npz`, `8_raw_forecasts_rolling_origin.npz` | — | The forecasts themselves, one array per `dataset|model|seed`. The tests in files 3 and 4 are computed from them. |

## Accuracy measures

For test values `a_t`, forecasts `f_t` and errors `e_t = a_t − f_t` over `T` test points:

| Measure | Formula | How to read it |
|---|---|---|
| MSE | mean of `e_t²` | The paper's main measure. It is in squared units of the series, so compare it only within one dataset. Lower is better. |
| MAD | mean of `|e_t|` | Average error size, in the units of the series; less affected by single large errors. |
| RMSE | `√MSE` | MSE back in the units of the series. |
| MAPE (%) | `100 · mean(|e_t| / |a_t|)` | Error as a percentage of the actual value; unreliable when actual values are near 0. |
| sMAPE (%) | `100 · mean(2|e_t| / (|a_t| + |f_t|))` | Symmetric percentage error; bounded and steadier than MAPE. |
| MASE | MAD ÷ mean `|y_t − y_{t−1}|` over the training part | Below 1: smaller errors than the naive forecast "same as the last value" makes on the training data. Unit-free. |
| TheilU2 | `√(MSE / MSE of the random walk)` on the test part | Below 1: better than forecasting "no change". |
| DA (%) | share of steps where `f_t − a_{t−1}` and `a_t − a_{t−1}` have the same sign | Direction accuracy; 50% is what guessing achieves. |

**Windows (column `window`):**
- "35 points ahead", "6 months" and similar are Zhang's windows: the average over the first `k` test points of
  one-step forecasts. They are not forecasts made `k` steps ahead.
- "all test points" uses the whole test part.
- "iterated h=3" and "iterated h=6" are true multi-step forecasts. Each forecast is fed back as an input until `h`
  steps ahead, on up to 30 starting points of the rolling protocol. Column `h` gives the horizon.

## Diebold–Mariano test (file 3)

The Diebold–Mariano (DM) test checks whether two forecasts of the same test points differ in accuracy by more than
chance.
- For each test point it takes the difference in squared error, `d_t = e_ref,t² − e_model,t²`. The reference is
  Zhang or ARIMA (column `versus`).
- The statistic is the mean of `d_t` divided by its standard error, `mean(d) / √(var(d)/T)`, multiplied by the
  Harvey–Leybourne–Newbold small-sample correction `√((T−1)/T)`.
- It is compared with a t distribution with `T−1` degrees of freedom. Forecasts are averaged over seeds first.

| Column | Reading |
|---|---|
| `dm_stat` | **Positive: the model has the lower error than the reference; negative: the reference is better.** |
| `p_value` | **Below 0.05: the difference is significant** (unlikely to be chance). Above 0.05: the test data cannot tell the two apart. |

## Model confidence set (file 4)

The model confidence set (MCS) keeps every model that cannot be distinguished from the best one at the 90% level.
It repeatedly removes the worst model until the remaining ones are statistically equivalent, using the range
statistic and a block bootstrap (blocks of 5, 1,000 replications). A model is in the set when `mcs_p` ≥ 0.10. A
large set means the test data cannot separate the models.
