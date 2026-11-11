# Website data schema

All files are strict JSON. There are no `NaN` or `Infinity` values; missing values are `null`, which
`tests_check_everything_works/test_website_data.py` checks. They are written by three scripts:

- `export_json.py` writes the per-dataset files and the result tables, from `results/`.
- `website_experiments.py` writes the five website experiments.
- `build_site_data.py` copies what the site uses into `research_site/data/`, filtering and averaging some tables.

Model names: `ARIMA`, `ANN`, `Zhang` (Zhang's hybrid), `Extension`, `Extension margin` (fixed protocol only), and
`E1`–`E7` (the variants). Protocols: `fixed` (Zhang's single split) and `rolling` (rolling origin with
validation-only tuning).

## Per-dataset files: `<dataset>.json` (11 files; in the site under `data/datasets/`)

| Field | Type | Meaning |
|---|---|---|
| `name`, `source`, `frequency`, `transform` | string | Dataset name, where it comes from, sampling frequency, transformation applied (`none`, `log10`, `sqrt`) |
| `n_train` | int | Length of the training part; the test part follows it |
| `series` | number[] | The whole series (training + test), after transformation |
| `actual_test` | number[] | The test part |
| `arima_forecast` | number[] | One-step ARIMA forecast of each test point (ARIMA fitted once on the training part) |
| `arima_error` | number[] | `actual_test − arima_forecast` |
| `models.<protocol>.<model>.forecast` | number[] | One-step test forecast, mean over seeds 0–9 |
| `models.<protocol>.<model>.forecast_seed_std` | number[] | Standard deviation of the forecast across seeds (0 for ARIMA) |
| `models.<protocol>.<model>.nn_correction` | number[] | `forecast − arima_forecast`: what the model adds to the ARIMA forecast |
| `models.<protocol>.<model>.final_error` | number[] | `actual_test − forecast` |

## Result tables (copies of `results/` CSV files; columns as described in each `results/*/README.md`)

| File | Source | Rows |
|---|---|---|
| `stationarity.json` | `results/1_stationarity/stationarity_tests.csv` | one per dataset |
| `validation_overview.json` | `results/2_model_comparison/1_test_mse_overview.csv` | protocol × dataset |
| `validation_tuning.json` | `.../2_chosen_hyperparameters.csv` | dataset × model |
| `validation_dm.json` | `.../3_significance_diebold_mariano.csv` | protocol × dataset × model × reference |
| `validation_mcs.json` | `.../4_model_confidence_set.csv` | protocol × dataset × model |
| `validation_summary.json` | `.../5_errors_mean_std_per_window.csv` | protocol × dataset × model × window |
| `nonlinearity_vs_improvement.json` | `results/3_residuals_and_decomposition/1_nonlinearity_vs_hybrid_gain.csv` | one per dataset |
| `decomposition_r2.json` | `.../2_correction_fit.csv` | dataset × model |
| `residual_diagnostics.json` | `.../3_residual_tests.csv` | dataset × residual set |
| `grid_when.json` | `results/4_alternative_hybrids_grid/1_best_hybrid_per_dataset.csv` | one per dataset |
| `grid_leaderboard.json` | `.../2_top10_hybrids_per_dataset.csv` | ten per dataset |
| `grid_single_models.json` | `.../3_single_model_baselines.csv` | dataset × model |
| `grid_results.json` | `.../4_raw_every_run.csv` | every run |

## Website experiments (`website_experiments.py`)

### `break_even.json`
Synthetic series `y_t = 0.6 y_{t−1} − 0.3 y_{t−2} + strength · sin(y_{t−1}) + noise · e_t`, with 240 training points
and 60 test points, and network sizes fixed in advance.

| Field | Meaning |
|---|---|
| `strengths`, `noises`, `seeds`, `models` | The grid: nonlinear strengths, noise standard deviations, series seeds, models compared |
| `network` | Lags, hidden units, error inputs and Extension restarts used |
| `cells[]` | One cell per strength and noise level; fields below |
| `cells[].strength`, `cells[].noise` | The cell's position on the grid |
| `cells[].mse_mean.<model>` | Test MSE, mean over seeds |
| `cells[].mse_by_seed.<model>` | Test MSE for each seed |
| `cells[].winner`, `cells[].runner_up` | Lowest and second-lowest mean MSE |
| `cells[].winner_beats_runner_up_in_seeds`, `cells[].n_seeds` | In how many seeds the winner beat the runner-up, out of `n_seeds` |
| `cells[].consistent` | True when the winner beat the runner-up in every seed |

### `form_lab.json`
A positive level `L_t = 10·exp(0.3·x_t)`, with `x` an AR(2), and `g_t = sin(y_{t−1} − 10)`:
- multiplicative: `y = L(1 + s·g) + v`;
- additive: `y = L + 10·s·g + v`;
- in both, `v ~ N(0, 0.2²)`.

| Field | Meaning |
|---|---|
| `strengths`, `forms`, `seeds`, `models` | The settings compared |
| `rows[]` | One row per form and strength; fields below |
| `rows[].form`, `rows[].strength` | The row's generating form and strength |
| `rows[].mse_mean`, `rows[].mse_std`, `rows[].mse_by_seed` | Test MSE per model: mean, standard deviation and per seed |
| `rows[].arima_constant_share` | Share of seeds in which ARIMA reduced to a constant forecast. In those cases E2 and Zhang give identical forecasts, because E2's ratio input is then a linear rescaling of Zhang's error input. |

### `rolling_animation.json`
`datasets.<Lynx|Nile>` holds, for each dataset:

| Field | Meaning |
|---|---|
| `series` | The whole series |
| `n_train` | Length of the training part |
| `test_index` | Position of each test point in the series |
| `refit_origins` | Positions at which the models were refitted |
| `fitted_through[i]` | End of the training window used for test point `i` |
| `forecasts.<ARIMA\|Zhang\|Extension>` | One-step forecasts, mean over seeds, from the saved rolling-protocol results |
| `n_seeds` | Number of seeds averaged for each model |

### `seed_fan.json`
`datasets.<Sunspot|Lynx|Synthetic additive>` holds:
- `actual_test` and `n_train`;
- `forecasts.<Zhang|Extension>.<seed>`: the fixed-split test forecast for each seed, 0–9.

### `scale_margin.json`
Sunspot, fixed split.

| Field | Meaning |
|---|---|
| `years` | Calendar year of each test point |
| `actual_test` | The test values |
| `train_min`, `train_max` | Smallest and largest training values |
| `margins`, `seeds` | Scale margins compared and seeds averaged |
| `record_points` | Test indices above the training maximum |
| `forecasts.<margin>.forecast` | Extension test forecast for that margin, mean over seeds |
| `forecasts.<margin>.mse`, `forecasts.<margin>.mse_record_points` | Test MSE over all test points and over the record points only |
| `forecasts.<margin>.max_forecast` | Largest forecast value |

## Site-only files (`build_site_data.py`, in `research_site/data/`)

| File | Content |
|---|---|
| `datasets_index.json` | One row per dataset: `name`, `slug` (file name), `source`, `frequency`, `transform`, `n_train`, `n_test`, `n_total` |
| `metrics.json` | `validation_summary` rows for the window "all test points": protocol, dataset, model and every `<measure>_mean` and `<measure>_std` |
| `dm.json` | Diebold–Mariano test of every model against Zhang and against ARIMA: `protocol`, `dataset`, `model`, `versus` (the reference), `dm_stat` (positive = lower error than the reference), `p_value` |
| `grid_designs.json` | Grid runs averaged over seeds: `dataset`, `linear`, `nonlinear`, `combiner`, `MSE`, `MSE_std`, `n_runs` |
| `residual_findings.json` | `{dataset: paragraph}` from `RESIDUAL_FINDINGS.md` |
