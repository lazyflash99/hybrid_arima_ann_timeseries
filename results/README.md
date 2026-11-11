# Results

Every table produced by the project. All files are written by the scripts; none is edited by hand. Rerun
everything with `python run_all.py --full`.

The folders follow the order of the analysis, and **each folder has its own README** that explains its files,
columns, measures and tests. Inside a folder the files are numbered: short summary tables come first, and raw
files with one row per run come last.

| Folder | Question it answers | Written by |
|---|---|---|
| [`1_stationarity/`](1_stationarity/README.md) | Is each series stationary, and how much differencing does ARIMA need? | `stationarity.py` |
| [`2_model_comparison/`](2_model_comparison/README.md) | Which model forecasts best, and is the difference significant? | `validation.py` |
| [`3_residuals_and_decomposition/`](3_residuals_and_decomposition/README.md) | What is left in ARIMA's errors, and how much of it does the network correct? | `residual_diagnostics.py`, `plots_decomposition.py` |
| [`4_alternative_hybrids_grid/`](4_alternative_hybrids_grid/README.md) | Do other linear and nonlinear models, and other ways of combining them, work better? | `hybrid_framework.py` |
| [`5_original_three_datasets/`](5_original_three_datasets/README.md) | Earlier experiments on the paper's three datasets | `run_extensions.py`, `tune_*.py`, `compare_*.py`, `make_extension_report.py` |
| `run_log.json` | Library versions and run time of each step of the last `run_all.py` run | `run_all.py` |
| `RESULTS_SUMMARY.md` (project root) | The main tables of folders 2–4 in one page | `make_results_summary.py` |

The CSV files open in Excel or any spreadsheet program. `python check_results.py` lists every empty cell in these
files with the reason it is empty.

## Terms used in every folder

- **Datasets:** the 11 series described in `DATASETS.md` (project root), which also says where each one is stored. Each is split in time into a training part and a test part.
- **Models:**
  - ARIMA: the linear model alone.
  - ANN: the neural network alone.
  - Zhang: ARIMA forecast plus a network forecast of ARIMA's error.
  - Extension: one network that receives past values, the ARIMA forecast and past ARIMA errors.
  - Extension margin: the Extension with a scale margin of 0.2.
  - E1–E7: other ways of combining the two parts.

  The main `README.md` describes each of them.
- **Seed:** the random starting point of a network. Networks are run with seeds 0–9 and usually averaged. ARIMA has
  no random element and is run once.
- **One-step forecast:** every forecast uses the actual past values, as in the paper.
