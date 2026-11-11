# 5. Original three datasets figures

Sunspot, Lynx and Exchange rate only, on the test part. Single run with seed 0 unless stated otherwise.
**Numbers:** `results/5_original_three_datasets/`.

| File | Written by | Content |
|---|---|---|
| `results_<dataset>.png` | `run_experiments.py` | Paper-style figure, as in Zhang's Figs. 4–6: actual values (solid) against forecast (dotted) for ARIMA, ANN, Zhang's hybrid and the Extension with scale margin, one panel each. |
| `results_extension_<dataset>.png` | `run_extension_experiments.py` | The same layout for ARIMA, ANN, Zhang's hybrid and the Extension without margin. |
| `extension_actual_vs_predicted_<dataset>.png` | `make_extension_report.py` | Actual against forecast for Zhang's hybrid, the Extension and the Extension + margin, one panel each. |
| `extension_overlay_<dataset>.png` | `make_extension_report.py` | The three forecasts on one chart with calendar years, for a direct comparison at peaks and troughs. |
| `extension_mse_mad_10_seeds.png` | `make_extension_report.py` | MSE and MAD per window, mean over seeds 0–9. The whisker on each bar is ± one standard deviation across seeds; a long whisker means the result depends on the random starting point. |
| `extension_mase_theil.png` | `make_extension_report.py` | MASE (below 1 beats the naive "last value" forecast) and Theil U2 (below 1 beats "no change") for each dataset, seed 0. |

**How to read the line charts:** where the forecast line lies on the actual line, the model is accurate. A forecast
that peaks one step after the actual series is mostly copying the last value.

**Why they are here:** they compare our reproduction with the paper's figures and show where the Extension differs
from Zhang's hybrid on the original data.
