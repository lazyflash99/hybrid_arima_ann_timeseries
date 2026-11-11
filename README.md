# Hybrid ARIMA + Neural Network Forecasting: reproduction and extensions of Zhang (2003)

Does combining ARIMA with a neural network forecast better, and when? This project:
- reproduces Zhang's hybrid model ("Time series forecasting using a hybrid ARIMA and neural network model",
  Neurocomputing 50:159-175, `nn_arima.pdf`);
- tests its central assumption that a series is a sum of a linear and a nonlinear part;
- compares it with an Extension and seven variants that combine the two parts differently.

All models are evaluated on 11 datasets, with validation-only tuning and significance tests.

DA 3041 course project, IIT Guwahati: Arnav Sinha, Arghya Ojha, Kanhav Purohit, Mohammad Shafqat Jabbar.

- Main tables: `RESULTS_SUMMARY.md`
- Every result file: `results/`
- Every figure: `figures/`
- Write-up: `report/main.pdf`

## Quickstart

Python 3.12:

    pip install -r requirements.txt
    python run_all.py --quick                     # 3 datasets, 2 seeds, small grids: a few minutes
    python run_all.py --full                      # every result and figure: about 54 minutes on 16 cores
    python -m pytest tests_check_everything_works # 66 tests, about 30 seconds

LightGBM and PyTorch are optional; without them the code uses RandomForest and skips the GRU. Each run writes
library versions and step times to `results/run_log.json`.

## The models

Notation:
- `y_t`: the series.
- `L̂_t`: the one-step ARIMA forecast.
- `e_t = y_t − L̂_t`: the ARIMA error.
- A "network" is the paper's single-hidden-layer network (logistic hidden units, linear output).

### Baselines from the paper

| Model | What it does | File |
|---|---|---|
| **ARIMA** | Linear model. The order is chosen automatically on the training data, as in the paper's Box–Jenkins procedure: the number of differences `d` from the ADF test, then the smallest model whose errors pass the Ljung–Box test. It is estimated by conditional least squares. | `arima_model.py` |
| **ANN** | The paper's network (Eq. 2) on past values `y_{t−1}, …, y_{t−p}`, trained by BFGS on the training MSE. | `ann_model.py` |
| **Zhang's hybrid** | Assumes `y_t = L_t + N_t`. ARIMA forecasts the linear part, and a network forecasts the ARIMA error from its past values: `ŷ_t = L̂_t + N̂_t`, with `N̂_t = f(e_{t−1}, …, e_{t−n})`. | `hybrid_model.py` |

### Our extensions

**Extension (EXT), our main model** (`extension_model.py`). One network produces the whole forecast from past
values, the ARIMA forecast and past ARIMA errors:

    ŷ_t = f(y_{t−1}, …, y_{t−p}, L̂_t, e_{t−1}, …, e_{t−q})

Zhang's hybrid always adds the network's correction to the ARIMA forecast. The Extension removes that fixed sum:
the network learns how much to trust ARIMA, and its correction can depend on the level of the series. Giving the
network `L̂_t` as an input is our own design. Training differs from the paper's network: L-BFGS-B, an L2 weight
penalty, and the best of several random starts (20 in the fixed protocol, 5 in the rolling protocol).

Options of the Extension:

| Option | What it does |
|---|---|
| **Scale margin** (`scale_margin=m`, "Extension + margin" with m = 0.2) | Maps the training range to `[m, 1−m]` instead of `[0, 1]`. Record values beyond the training range then stay away from the flat ends of the logistic units. |
| **Skip output** (`output="skip"`) | The network outputs only a correction that is added to `L̂_t`, so the ARIMA forecast reaches the output unchanged. |
| **Shortcut** (`shortcut=True`) | Adds direct linear connections from every input to the output, alongside the hidden layer. |

**Variants E1–E7** (`extensions.py`). Each tries another way of combining ARIMA and the network. All use the
automatically chosen ARIMA and the paper's network unless stated otherwise.

| Variant | Formula | What it changes |
|---|---|---|
| **E1 Skip combination** | `ŷ_t = L̂_t + f(y lags, L̂_t, e lags)` | The Extension's inputs, but the network outputs a correction added to the ARIMA forecast. |
| **E2 Multiplicative hybrid** | `ŷ_t = L̂_t · N̂_t`, with `N = y / L` | Assumes the nonlinear part multiplies the linear part instead of adding to it. The network forecasts the ratio of actual to ARIMA forecast. |
| **E3 Wavelet decomposition** | `y = A + D` | A causal Haar wavelet splits the series into a smooth part `A`, modelled by ARIMA, and a detail part `D`, modelled by the network. The forecasts are added. The split is by frequency, not by a linear-plus-nonlinear assumption. |
| **E4 Gated hybrid** | `ŷ_t = L̂_t + G_t · N̂_t`, with `G_t ∈ (0, 1)` | A learned gate that depends on the recent level and on `L̂_t` decides how much of the network correction to use at each step. |
| **E5 Joint AR + network** | `ŷ_t = AR(y lags) + f(y lags)` | The AR coefficients and the network weights are estimated together, not in two stages. The linear part must be a pure AR model; if ARIMA selects MA terms, E5 fits its own AR model. |
| **E6 Weighted combination** | `ŷ_t = c + a·L̂_t + b·N̂_t + g·ANN_t` | Combines the ARIMA forecast, Zhang's correction and the network alone with weights estimated by least squares. Zhang's hybrid is the case `a = b = 1`. |
| **E7 Validated combination** | as E6 | The weights are estimated on forecasts for the last 20% of the training data, which the base models were not fitted on, so the networks are not over-weighted. |

**Generic hybrid framework** (`hybrid_framework.py`). Every hybrid is a combination of three parts:
- **Linear model:** ARIMA, SARIMA, ETS, Theta, OLS AR, naive or seasonal naive.
- **Nonlinear model:** the network, SVR, LightGBM or GRU.
- **Combiner:** additive, multiplicative, joint-input or validation-weighted.

All 105 valid designs are tested on every dataset.

## How the models are evaluated

| Protocol | Description |
|---|---|
| **Fixed split** | Zhang's protocol: one fit on the training part with network sizes fixed in advance, then one-step forecasts on the test part. |
| **Rolling origin** | Settings (hidden units, lags, margin, weight decay) are chosen on validation data inside the training part. The model is then refitted 3 times as it moves through the test part. No decision uses test data. |

Every network is run with seeds 0–9. Differences are tested with the Diebold–Mariano test and the model confidence
set (explained in `results/2_model_comparison/README.md`).

## Datasets

The 11 series:
- **Zhang's three:** Sunspot, Lynx and the British pound exchange rate.
- **Five public series:** AirPassengers, Nile, Melbourne daily minimum temperature, the Japanese yen exchange rate,
  and a square-root sunspot series.
- **Three synthetic series with known structure:** additive, multiplicative and Mackey-Glass.

**Where they are:**
- Five are CSV files in `data/`.
- Three come with the statsmodels package.
- Three are generated by `data_loader.py` with a fixed seed.

`DATASETS.md` lists each dataset's storage location, source, length, train/test split, transformation and seasonal
period. `data_loader.DATASETS` is the registry used by the code.

## Repository layout

| Path | Content |
|---|---|
| `run_all.py` | Runs every step in order and writes `results/run_log.json` |
| `config.py` | Paths, seeds, run modes, tuning grids, and the names of the result folders |
| `data/`, `data_loader.py` | Dataset files and the loader and registry |
| `arima_model.py`, `ann_model.py`, `hybrid_model.py` | The paper's models |
| `extension_model.py`, `extensions.py`, `hybrid_framework.py` | The Extension, variants E1–E7, and the generic framework with its grid |
| `stationarity.py` | U6: stationarity tests and figures |
| `validation.py` | U4: both protocols, Diebold–Mariano, model confidence set |
| `plots_decomposition.py` | U2: forecast decomposition figures |
| `residual_diagnostics.py` | U3: residual tests and figures |
| `run_experiments.py`, `run_extension_experiments.py`, `run_extensions.py`, `tune_*.py`, `compare_*.py`, `make_*.py` | Experiments, tuning and figures on the paper's three datasets |
| `make_results_summary.py`, `check_results.py` | Writes `RESULTS_SUMMARY.md`; lists every empty value in the results with its reason |
| `results/`, `figures/` | All outputs, in numbered topic folders, each with a README |
| `tests_check_everything_works/` | Automated tests: no forecast uses future data, results are reproducible, every model builds and forecasts |
| `report/`, `presentation/` | The ICML-format report and the slides |
| `website/` | Data for a future project website, kept apart from the code |
| `AUDIT.md`, `REFERENCES.md`, `RESIDUAL_FINDINGS.md` | Audit of design decisions, references, and the residual findings per dataset |

## How each update is reproduced

| Update | Command | Output |
|---|---|---|
| U1 More datasets | `python -m pytest tests_check_everything_works/test_data.py` | `DATASETS.md` |
| U2 Decomposition plots | `python plots_decomposition.py` | `figures/2_forecast_decomposition/`, `results/3_residuals_and_decomposition/` |
| U3 Residual diagnostics | `python residual_diagnostics.py` | `results/3_residuals_and_decomposition/`, `figures/3_residual_diagnostics/`, `RESIDUAL_FINDINGS.md` |
| U4 Validation | `python validation.py` | `results/2_model_comparison/` |
| U5 Alternatives to ARIMA and the ANN | `python hybrid_framework.py` | `results/4_alternative_hybrids_grid/`, `figures/4_alternative_hybrids_grid/` |
| U6 Stationarity | `python stationarity.py` | `results/1_stationarity/`, `figures/1_stationarity/` |

Add `--quick` to any of the last five commands for a small run.

## Percentage improvement

The paper's improvement figures use the hybrid's error as the denominator:
`(error_other − error_hybrid) / error_hybrid × 100`.

## License

MIT, see `LICENSE`.
