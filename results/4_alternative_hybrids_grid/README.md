# 4. Alternative hybrids grid

**Question:** does the hybrid idea work better with other components than ARIMA and the logistic network, or with
another way of combining them?
**Written by:** `hybrid_framework.py`, using the rolling protocol (see `../2_model_comparison/README.md`).
**Figures:** `figures/4_alternative_hybrids_grid/`.

A hybrid design is a combination of three parts.

| Part | Options |
|---|---|
| Linear model | ARIMA, SARIMA, ETS (Holt–Winters), Theta, OLS AR, Naive (last value), Seasonal naive (value one season ago) |
| Nonlinear model | MLP (the paper's network), SVR (support vector regression), LightGBM (gradient-boosted trees), GRU (small recurrent network) |
| Combiner | **additive:** linear forecast + nonlinear model of the linear error (Zhang's form)<br>**multiplicative:** linear forecast × nonlinear model of the ratio actual / linear forecast<br>**joint-input:** one nonlinear model receives past values, the linear forecast and past linear errors (the Extension's form)<br>**validation-weighted:** linear forecast + w × nonlinear correction, with w between 0 and 1 chosen on validation data |

That gives 7 × 4 × 4 = 112 combinations, minus GRU with joint input, which is not defined: 105 designs per
dataset. Each design chooses its number of past values (2, 4 or 7) on one validation fold inside the training part.
It is then run with seeds 0–9; SVR has no random element and is run once.

## Files

| File | Rows | Content |
|---|---|---|
| `1_best_hybrid_per_dataset.csv` | 11 | **Start here.** One row per dataset; columns below. |
| `2_top10_hybrids_per_dataset.csv` | 110 | The ten designs with the lowest mean test MSE per dataset, with `rank`, `relative_to_best_single` (MSE ÷ MSE of the best single model; below 1 = the hybrid is better), `best_single`, and a Diebold–Mariano test against that single model (`dm_stat`, `dm_p`). |
| `3_single_model_baselines.csv` | 121 | Each linear model and each nonlinear model used alone (`kind`), with all eight accuracy measures. |
| `4_raw_every_run.csv` | 8,778 | Every run: dataset, design, the number of past values chosen (`lags`), seed, status, and all eight accuracy measures. |

The accuracy measures are defined in `../2_model_comparison/README.md`.

### Columns of `1_best_hybrid_per_dataset.csv`

| Column | Meaning |
|---|---|
| `best_hybrid`, `best_hybrid_mse` | Design with the lowest mean test MSE, written "linear + nonlinear (combiner)", and that MSE. |
| `best_single`, `best_single_mse` | Single model (linear or nonlinear alone) with the lowest test MSE, and that MSE. |
| `simple_baseline_wins` | True when the best single model is at least as good as the best hybrid. |
| `seasonal_strength` | Strength of seasonality of the training series, copied from `../1_stationarity/stationarity_tests.csv`. |
| `nonlinearity_detected` | Whether nonlinearity was found in ARIMA's errors, copied from `../3_residuals_and_decomposition/1_nonlinearity_vs_hybrid_gain.csv`. |

**How seasonal strength is calculated:**
1. The training series is split with an STL decomposition (seasonal-trend decomposition by Loess, at the dataset's
   seasonal period) into trend, seasonal part `S` and remainder `R`.
2. Then

       seasonal strength = max(0, 1 − Var(R) / Var(S + R))

   This is Hyndman's measure F_s.

A value near 1 means the seasonal pattern dominates the remainder; near 0 means little seasonality. It is empty
when the dataset has no seasonal period or is shorter than two full periods. The last two columns help relate
"which hybrid works" to properties of the data.

## Diebold–Mariano test

The Diebold–Mariano (DM) test checks whether two forecasts of the same test points differ in accuracy by more than
chance.
- With `d_t` = squared error of the best single model minus squared error of the hybrid, the statistic is
  `mean(d) / √(var(d)/T)` with the Harvey–Leybourne–Newbold small-sample correction.
- It is compared with a t distribution with `T−1` degrees of freedom.
- **`dm_stat` > 0: the hybrid has the lower error. `dm_p` < 0.05: the difference is significant.**

**Caution:** the "best" hybrid is picked on test error from 105 candidates, so its advantage is optimistic. The DM
p-value shows whether it is real.
