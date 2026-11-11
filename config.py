"""
Central settings: paths, seeds, run modes and the hyperparameter grids searched on validation data.
The dataset registry itself (sources, splits, transformations) is data_loader.DATASETS.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")
FIGURES_DIR = os.path.join(ROOT, "figures")
WEBSITE_DATA_DIR = os.path.join(ROOT, "website", "data")
WORKERS = max(1, min(12, (os.cpu_count() or 2) - 2))

# Where each result file is stored. Scripts refer to a file by its short name; this map places it in a topic
# folder. Within a folder, the numbering puts short summary tables first and raw run-level files last.
RESULT_FILES = {
    "stationarity.csv": "1_stationarity/stationarity_tests.csv",
    "validation_overview.csv": "2_model_comparison/1_test_mse_overview.csv",
    "validation_tuning.csv": "2_model_comparison/2_chosen_hyperparameters.csv",
    "validation_dm.csv": "2_model_comparison/3_significance_diebold_mariano.csv",
    "validation_mcs.csv": "2_model_comparison/4_model_confidence_set.csv",
    "validation_summary.csv": "2_model_comparison/5_errors_mean_std_per_window.csv",
    "validation_results.csv": "2_model_comparison/6_raw_every_run.csv",
    "forecasts_fixed.npz": "2_model_comparison/7_raw_forecasts_fixed_split.npz",
    "forecasts_rolling.npz": "2_model_comparison/8_raw_forecasts_rolling_origin.npz",
    "nonlinearity_vs_improvement.csv": "3_residuals_and_decomposition/1_nonlinearity_vs_hybrid_gain.csv",
    "decomposition_r2.csv": "3_residuals_and_decomposition/2_correction_fit.csv",
    "residual_diagnostics.csv": "3_residuals_and_decomposition/3_residual_tests.csv",
    "grid_when.csv": "4_alternative_hybrids_grid/1_best_hybrid_per_dataset.csv",
    "grid_leaderboard.csv": "4_alternative_hybrids_grid/2_top10_hybrids_per_dataset.csv",
    "grid_single_models.csv": "4_alternative_hybrids_grid/3_single_model_baselines.csv",
    "grid_results.csv": "4_alternative_hybrids_grid/4_raw_every_run.csv",
    "extension_metrics_seed0.csv": "5_original_three_datasets/1_single_run_metrics_seed0.csv",
    "results_extensions.csv": "5_original_three_datasets/2_all_models_10_seeds.csv",
    "results_margin_tuning.csv": "5_original_three_datasets/3_scale_margin_tuning.csv",
    "results_error_inputs.csv": "5_original_three_datasets/4_past_error_inputs_tuning.csv",
    "results_past_values.csv": "5_original_three_datasets/5_past_value_inputs_tuning.csv",
    "results_no_lhat.csv": "5_original_three_datasets/6_without_arima_forecast_input.csv",
    "results_arima_lags.csv": "5_original_three_datasets/7_inputs_at_arima_lags.csv",
}

# Figure folders, in the order they are discussed in the report.
FIGURE_DIRS = {
    "stationarity": "1_stationarity",
    "decomposition": "2_forecast_decomposition",
    "residuals": "3_residual_diagnostics",
    "grid": "4_alternative_hybrids_grid",
    "paper": "5_original_three_datasets",
    "extension_report": "5_original_three_datasets",
    "slides": "6_presentation_slides",
}


def results_path(name):
    """Full path of a result file, given its short name (see RESULT_FILES)."""
    path = os.path.join(RESULTS_DIR, RESULT_FILES.get(name, name))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def figure_path(group, name):
    """Full path of a figure file in the folder of its group (see FIGURE_DIRS)."""
    path = os.path.join(FIGURES_DIR, FIGURE_DIRS.get(group, group), name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


SEEDS_FULL = list(range(10))
SEEDS_QUICK = [0, 1]
VAL_FRAC = 0.2                  # the last part of the training data used for validation

# settings of the two run modes (run_all.py --quick / --full)
MODES = {
    "quick": dict(
        seeds=SEEDS_QUICK, val_folds=1, n_refits=1, restarts_tune=1, restarts_final=2,
        grid=dict(hidden=[2, 4], lags=[2, 4], margin=[0.0, 0.2], weight_decay=[1e-4]),
        datasets=["Sunspot", "AirPassengers", "Synthetic additive"],
        multi_h=[3], max_iter=200,
        fw_seeds=[0], fw_lin=["ARIMA", "OLS AR", "Naive"], fw_nl=["MLP", "SVR"],
        fw_comb=["additive", "joint-input"], fw_grid=dict(lags=[2, 4]), mcs_boot=200),
    "full": dict(
        seeds=SEEDS_FULL, val_folds=2, n_refits=3, restarts_tune=2, restarts_final=5,
        grid=dict(hidden=[2, 4, 6], lags=[2, 4, 7], margin=[0.0, 0.1, 0.2], weight_decay=[1e-5, 1e-3]),
        datasets=None,          # every dataset in the registry
        multi_h=[3, 6], max_iter=2000,
        fw_seeds=SEEDS_FULL, fw_lin=["ARIMA", "SARIMA", "ETS", "Theta", "OLS AR", "Naive", "Seasonal naive"],
        fw_nl=["MLP", "SVR", "LightGBM", "GRU"],
        fw_comb=["additive", "multiplicative", "joint-input", "validation-weighted"],
        fw_grid=dict(lags=[2, 4, 7]), mcs_boot=1000),
}
