"""
Zhang's hybrid, the Extension and the Extension with scale margin on the three original datasets: errors, extra
accuracy measures and plots.   Usage:  python make_extension_report.py
Writes figures/5_original_three_datasets/extension_*.png and results/5_original_three_datasets/1_single_run_metrics_seed0.csv.

  * single run (seed 0, as in the paper's tables): MSE / MAD per horizon and extra measures on the full test set
  * mean +- std over seeds 0-9 from results_extensions.csv (run run_extensions.py first) for the MSE / MAD bars

Extra measures (full test set):
  RMSE           square root of MSE, in the units of the series
  MAPE (%)       mean |error| / |actual|; unstable when actual values are near 0 (sunspot minima)
  sMAPE (%)      symmetric MAPE, 2|error| / (|actual| + |forecast|); bounded, less sensitive to small actuals
  MASE           MAD / MAD of the naive forecast y_hat_t = y_{t-1} on the training data; < 1 beats naive
  Theil's U2     RMSE / RMSE of the random walk on the test set; < 1 beats "no change"
  DA (%)         directional accuracy: share of steps where the forecast change has the sign of the actual change
  DM vs Zhang    Diebold-Mariano test (squared error, one step, Harvey-Leybourne-Newbold correction);
                 positive = lower error than Zhang's hybrid, p < 0.05 = the difference is significant
"""
import os
import numpy as np
from config import results_path, figure_path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t

from arima_model import AutoARIMA
from hybrid_model import HybridModel
from extension_model import ExtensionHybrid
from run_experiments import EXPERIMENTS

OUT = os.path.dirname(figure_path("extension_report", "x"))
SEED = 0
MODELS = ["Zhang hybrid", "Extension", "Extension + margin"]
CSV_NAMES = {"Zhang hybrid": "Zhang hybrid", "Extension": "Extension",
             "Extension + margin": "Extension + margin"}
# one colour per model, used in every figure
COLORS = {"Zhang hybrid": "#2a78d6", "Extension": "#eb6834", "Extension + margin": "#1baf7a"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def forecasts(cfg, y_train, y):
    p, q = cfg["ann"]
    n = len(y_train)
    ext = dict(cfg["ext_kwargs"], n_restarts=20, seed=SEED)
    models = {
        "Zhang hybrid": HybridModel(AutoARIMA(verbose=False), p, q, seed=SEED),
        "Extension": ExtensionHybrid(AutoARIMA(verbose=False), **ext),
        "Extension + margin": ExtensionHybrid(AutoARIMA(verbose=False), scale_margin=0.2, **ext),
    }
    return {name: m.fit(y_train).predict_one_step(y)[n:] for name, m in models.items()}


def extra_measures(y_train, y, n, f, f_ref):
    a, prev = y[n:], y[n - 1:-1]                       # actual values and the value one step before each
    err = a - f
    naive_mad_train = np.mean(np.abs(np.diff(y_train)))
    out = {
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "MAPE (%)": 100 * np.mean(np.abs(err) / np.abs(a)),
        "sMAPE (%)": 100 * np.mean(2 * np.abs(err) / (np.abs(a) + np.abs(f))),
        "MASE": np.mean(np.abs(err)) / naive_mad_train,
        "Theil U2": np.sqrt(np.mean(err ** 2)) / np.sqrt(np.mean((a - prev) ** 2)),
        "DA (%)": 100 * np.mean(np.sign(f - prev) == np.sign(a - prev)),
    }
    if f_ref is not None:                               # Diebold-Mariano vs Zhang's hybrid
        d = (a - f_ref) ** 2 - err ** 2
        m = len(d)
        dm = np.mean(d) / np.sqrt(np.var(d, ddof=1) / m) * np.sqrt((m - 1) / m)
        out["DM vs Zhang"] = dm
        out["DM p-value"] = 2 * student_t.sf(abs(dm), m - 1)
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_actual_vs_predicted(name, cfg, y_test, fc):
    """Paper style (Figs. 4-6): actual solid, prediction dotted, one panel per model."""
    spec = {"Sunspot": dict(ylim=(0, 250), step=3), "Lynx": dict(ylim=(0.0, 4.0), step=1),
            "Exchange rate": dict(ylim=(0.10, 0.22), step=3)}[name]
    x = np.arange(1, len(y_test) + 1)
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(7, 8.2))
    for i, (ax, model) in enumerate(zip(axes, MODELS)):
        ax.plot(x, y_test, "k-", linewidth=1.2, label="actual")
        ax.plot(x, fc[model], "k:", linewidth=1.4, dashes=(1, 2), label="prediction")
        ax.set_title(f"({'abc'[i]}) {model} prediction of {name.lower()}", fontsize=10, loc="left", pad=4)
        ax.set_xlim(1, len(y_test))
        ax.set_ylim(spec["ylim"])
        ax.set_xticks(list(range(1, len(y_test) + 1, spec["step"])))
        ax.tick_params(direction="in", top=True, right=True, labelsize=7)
    axes[-1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=True, fontsize=9)
    fig.subplots_adjust(hspace=0.45, bottom=0.11, top=0.96, left=0.12, right=0.95)
    path = os.path.join(OUT, f"extension_actual_vs_predicted_{name.replace(' ', '_').lower()}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_peak_overlay(name, y_test, fc, n_train):
    """All three forecasts on one chart, to compare them directly."""
    x = np.arange(n_train + 1, n_train + len(y_test) + 1)
    years = {"Sunspot": 1700, "Lynx": 1821}.get(name)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    style(ax)
    xs = x - 1 + years if years else np.arange(1, len(y_test) + 1)
    ax.plot(xs, y_test, color=INK, linewidth=2, label="actual")
    for model in MODELS:
        ax.plot(xs, fc[model], color=COLORS[model], linewidth=1.6, label=model)
    ax.set_xlabel("year" if years else "week of 1993", color=INK_2, fontsize=9)
    ax.set_title(f"{name}: actual vs one-step forecasts (seed {SEED})", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.18))
    fig.subplots_adjust(bottom=0.3, top=0.9, left=0.08, right=0.98)
    path = os.path.join(OUT, f"extension_overlay_{name.replace(' ', '_').lower()}.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_mse_mad(seed_df):
    """Mean +- std over seeds 0-9, one panel per data set and measure (no shared axis between measures)."""
    datasets = list(EXPERIMENTS)
    fig, axes = plt.subplots(len(datasets), 2, figsize=(10, 9))
    width = 0.26
    for r, name in enumerate(datasets):
        part = seed_df[seed_df.dataset == name]
        horizons = list(EXPERIMENTS[name]["horizons"])
        scale = 1e5 if name == "Exchange rate" else 1.0
        for c, measure in enumerate(("MSE", "MAD")):
            ax = axes[r, c]
            style(ax)
            cols = [f"{h} {measure}" for h in horizons]
            k = scale if measure == "MSE" else 1.0
            for j, model in enumerate(MODELS):
                rows = part[part.model == CSV_NAMES[model]][cols]
                ax.bar(np.arange(len(cols)) + (j - 1) * width, rows.mean() * k, width, yerr=rows.std() * k,
                       color=COLORS[model], edgecolor=SURFACE, linewidth=2, label=model,
                       error_kw=dict(ecolor=INK_2, elinewidth=1, capsize=2))
            ax.set_xticks(np.arange(len(cols)))
            ax.set_xticklabels(horizons, fontsize=8, color=INK_2)
            unit = " (x 1e-5)" if (measure == "MSE" and scale != 1) else ""
            ax.set_title(f"{name}: {measure}{unit}", color=INK, fontsize=10, loc="left")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.32))
    fig.suptitle("Test error, mean ± std over 10 seeds (lower is better)", x=0.06, ha="left", color=INK, fontsize=11)
    fig.subplots_adjust(hspace=0.5, wspace=0.22, top=0.88, bottom=0.05, left=0.07, right=0.98)
    path = os.path.join(OUT, "extension_mse_mad_10_seeds.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_scale_free(extra):
    """MASE and Theil's U2 are unit-free, so the three data sets share one chart per measure."""
    datasets = list(EXPERIMENTS)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    width = 0.26
    for ax, measure, ref in zip(axes, ("MASE", "Theil U2"), ("naive forecast", "random walk")):
        style(ax)
        for j, model in enumerate(MODELS):
            vals = [extra[(d, model)][measure] for d in datasets]
            ax.bar(np.arange(len(datasets)) + (j - 1) * width, vals, width, color=COLORS[model],
                   edgecolor=SURFACE, linewidth=2, label=model)
        ax.axhline(1.0, color=INK_2, linewidth=1, linestyle="--")
        ax.text(len(datasets) - 0.55, 1.02, f"= {ref}", color=INK_2, fontsize=8, ha="right", va="bottom")
        ax.set_xticks(np.arange(len(datasets)))
        ax.set_xticklabels(datasets, fontsize=9, color=INK_2)
        ax.set_title(f"{measure} (seed {SEED}, full test set; below 1 is better than the {ref})",
                     color=INK, fontsize=9, loc="left")
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.subplots_adjust(bottom=0.25, top=0.88, left=0.06, right=0.98, wspace=0.18)
    path = os.path.join(OUT, "extension_mase_theil.png")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rows, extra, written = [], {}, []
    for name, cfg in EXPERIMENTS.items():
        y = cfg["load"]()
        n = cfg["n_train"]
        y_train, y_test = y[:n], y[n:]
        fc = forecasts(cfg, y_train, y)
        scale = 1e5 if name == "Exchange rate" else 1.0
        print(f"\n===== {name} (seed {SEED}{', MSE x 1e5' if scale != 1 else ''}) =====")
        for model in MODELS:
            line, row = f"{model:20}", {"dataset": name, "model": model}
            for h_name, h in cfg["horizons"].items():
                e = y_test[:h] - fc[model][:h]
                row[f"{h_name} MSE"], row[f"{h_name} MAD"] = np.mean(e ** 2), np.mean(np.abs(e))
                line += f"  {h_name}: MSE {row[f'{h_name} MSE'] * scale:9.5g}  MAD {row[f'{h_name} MAD']:9.5g}"
            ref = None if model == "Zhang hybrid" else fc["Zhang hybrid"]
            extra[(name, model)] = extra_measures(y_train, y, n, fc[model], ref)
            row.update(extra[(name, model)])
            rows.append(row)
            print(line)
        print(f"  {'':20}" + "".join(f"{k:>13}" for k in extra[(name, MODELS[-1])]))
        for model in MODELS:
            print(f"  {model:20}" + "".join(f"{v:13.4g}" for v in extra[(name, model)].values()))
        written += [plot_actual_vs_predicted(name, cfg, y_test, fc), plot_peak_overlay(name, y_test, fc, n)]
    pd.DataFrame(rows).to_csv(results_path("extension_metrics_seed0.csv"), index=False)

    if os.path.exists(results_path("results_extensions.csv")):
        seed_df = pd.read_csv(results_path("results_extensions.csv"))
        written.append(plot_mse_mad(seed_df))
    written.append(plot_scale_free(extra))
    print("\nWritten:", *written, results_path("extension_metrics_seed0.csv"), sep="\n  ")
