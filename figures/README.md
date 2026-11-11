# Figures

Every figure produced by the project. Rerun everything with `python run_all.py --full`. The folders follow the
order of the analysis and match the folders in `results/`. **Each folder has its own README** explaining what each
figure shows, how to read it and why it is included.

| Folder | Files | What the figures show | Written by |
|---|---|---|---|
| [`1_stationarity/`](1_stationarity/README.md) | 11 | Each training series before and after differencing, with ACF and PACF | `stationarity.py` |
| [`2_forecast_decomposition/`](2_forecast_decomposition/README.md) | 33 | How the forecast splits into the ARIMA forecast and the neural-network correction | `plots_decomposition.py` |
| [`3_residual_diagnostics/`](3_residual_diagnostics/README.md) | 33 | What is left in the forecast errors | `residual_diagnostics.py` |
| [`4_alternative_hybrids_grid/`](4_alternative_hybrids_grid/README.md) | 11 | Test error of all 105 hybrid designs, one heatmap per dataset | `hybrid_framework.py` |
| [`5_original_three_datasets/`](5_original_three_datasets/README.md) | 14 | Forecasts and errors on the paper's three datasets | `run_experiments.py`, `run_extension_experiments.py`, `make_extension_report.py` |
| [`6_presentation_slides/`](6_presentation_slides/README.md) | 7 | Large-font versions for the slides | `make_slide_figures.py`, `make_error_figure.py` |

File names start with the dataset name in lower case. In every test-set plot, the x-axis counts test points from 1
to the length of the test part.

## Words used in the figures

- **ARIMA forecast:** the forecast of the linear model alone.
- **ARIMA error:** actual value minus ARIMA forecast.
- **Neural-network correction:** what a hybrid adds to the ARIMA forecast, i.e. its final forecast minus the ARIMA
  forecast.
- **Final forecast:** the forecast of the model named in the title.
- **Dashed red lines in ACF and PACF panels:** ±1.96/√n. Bars outside them differ significantly from zero at about
  the 5% level.
