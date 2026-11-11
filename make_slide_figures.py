"""
Figures for the presentation, written to figures/6_presentation_slides/ (presentation/images/ is not touched). The extensions are labelled
"Extension 1" (one combining network) and "Extension 2" (plus scale margin) throughout.   Usage:  python make_slide_figures.py
Needs results_extensions.csv (python run_extensions.py) for the 10-seed bar chart.
"""
import os
import numpy as np
from config import figure_path, results_path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from arima_model import AutoARIMA
from hybrid_model import HybridModel
from extension_model import ExtensionHybrid
from run_experiments import EXPERIMENTS

OUT = os.path.dirname(figure_path("slides", "x"))
EXT, EXT_M, HYB = "Extension 1", "Extension 1 Improved", "Hybrid (Zhang)"
CSV_NAMES = {HYB: "Zhang hybrid", EXT: "Extension", EXT_M: "Extension + margin"}
COLORS = {HYB: "#2a78d6", EXT: "#eb6834", EXT_M: "#1baf7a"}     # one colour per model
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
START_YEAR = {"Sunspot": 1700, "Lynx": 1821}
SPEC = {"Sunspot": dict(ylim=(0, 250), step=3), "Lynx": dict(ylim=(0.0, 4.0), step=1),
        "Exchange rate": dict(ylim=(0.10, 0.22), step=3)}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("  wrote", path)


def extension(cfg, seed, margin):
    ext = dict(cfg["ext_kwargs"], n_restarts=20, seed=seed)
    return ExtensionHybrid(AutoARIMA(verbose=False), scale_margin=margin, **ext)


# ------------------------------------------------------------------ large-font versions for the presentation
# Drawn at their physical size on a 4:3 slide so that 9-11 pt text remains legible.
def data_row():
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.4))
    labels = {"Sunspot": "Sunspot (yearly)", "Lynx": "Lynx, log10 (yearly)", "Exchange rate": "GBP/USD, log10 (weekly)"}
    for ax, (name, cfg) in zip(axes, EXPERIMENTS.items()):
        style(ax)
        ax.tick_params(labelsize=7)
        y, n = cfg["load"](), cfg["n_train"]
        x = (np.arange(len(y)) + START_YEAR[name]) if name in START_YEAR else 1980 + np.arange(len(y)) / 52.18
        ax.plot(x[:n], y[:n], color=INK, linewidth=0.8)
        ax.plot(x[n - 1:], y[n - 1:], color=COLORS[HYB], linewidth=1)
        ax.axvspan(x[n], x[-1], color=COLORS[HYB], alpha=0.1, linewidth=0)
        ax.set_title(labels[name], fontsize=9, loc="left", color=INK)
        ax.set_xlim(x[0], x[-1])
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    axes[0].plot([], [], color=INK, label="training")
    axes[0].plot([], [], color=COLORS[HYB], label="test")
    fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.subplots_adjust(wspace=0.3, top=0.88, bottom=0.27, left=0.06, right=0.99)
    save(fig, "big_data_row.png")


def big_overlay(y, n, fc):
    a = y[n:]
    years = np.arange(n, len(y)) + START_YEAR["Sunspot"]
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    style(ax)
    ax.tick_params(labelsize=8)
    ax.axhline(y[:n].max(), color=INK_2, linewidth=1, linestyle="--")
    ax.text(1962, y[:n].max() + 6, "highest value in training data (154.4)", color=INK_2, fontsize=8)
    ax.plot(years, a, color=INK, linewidth=2, label="actual")
    for label in (HYB, EXT, EXT_M):
        ax.plot(years, fc[label], label=label, **LINE_STYLE[label])
    ax.set_xlim(1945, 1987)
    ax.set_ylim(0, 212)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper left", bbox_to_anchor=(-0.01, -0.1),
              handlelength=1.5, columnspacing=1.2)
    fig.subplots_adjust(bottom=0.24, top=0.97, left=0.07, right=0.99)
    save(fig, "big_sunspot_overlay.png")


def series_x(name, n_total, n_train):
    if name in START_YEAR:
        return np.arange(n_train, n_total) + START_YEAR[name], "year"
    return np.arange(1, n_total - n_train + 1), "week of 1993"


# distinct line types, so the three models can be told apart without colour
LINE_STYLE = {HYB: dict(color="#1f4e9c", linestyle="-", linewidth=1.5),
              EXT: dict(color="#d95f02", linestyle=(0, (5, 2)), linewidth=1.6),
              EXT_M: dict(color="#0f8a5f", linestyle=(0, (2.2, 1.3)), linewidth=1.8)}


def big_lines(all_fc, labels, file_name, height=4.5, styled=False):
    """One panel per data set: actual and the chosen forecasts on the same axes.
    styled=True: actual as a thick light-grey line underneath, each model with its own line type on top."""
    fig, axes = plt.subplots(3, 1, figsize=(6.6, height))
    for ax, (name, (y, n, fc)) in zip(axes, all_fc.items()):
        style(ax)
        ax.tick_params(labelsize=7.5)
        x, _ = series_x(name, len(y), n)
        if styled:
            ax.plot(x, y[n:], color="#b9b8b2", linewidth=3.6, solid_capstyle="round", label="actual")
        else:
            ax.plot(x, y[n:], color=INK, linewidth=1.8, label="actual")
        for label in labels:
            kw = LINE_STYLE[label] if styled else dict(color=COLORS[label], linewidth=1.3)
            ax.plot(x, fc[label], label=label, **kw)
        ax.set_xlim(x[0], x[-1])
        ax.set_title(name, fontsize=9, loc="left", color=INK, pad=2)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8.5, ncol=1 + len(labels),
               loc="lower center", handlelength=2.6 if styled else 1.6)
    fig.subplots_adjust(hspace=0.55, top=0.95, bottom=0.12, left=0.07, right=0.96)
    save(fig, file_name)


def big_overlay_ext1(y, n, fc):
    a = y[n:]
    years = np.arange(n, len(y)) + START_YEAR["Sunspot"]
    fig, ax = plt.subplots(figsize=(6.6, 2.8))
    style(ax)
    ax.tick_params(labelsize=8)
    ax.axvspan(1956.5, 1961.5, color=COLORS[EXT], alpha=0.10, linewidth=0)
    ax.axhline(y[:n].max(), color=INK_2, linewidth=1, linestyle="--")
    ax.text(1962, y[:n].max() + 6, "highest value in training data (154.4)", color=INK_2, fontsize=8)
    ax.plot(years, a, color=INK, linewidth=2, label="actual")
    for label in (HYB, EXT):
        ax.plot(years, fc[label], label=label, **LINE_STYLE[label])
    ax.set_xlim(1945, 1987)
    ax.set_ylim(0, 212)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left", bbox_to_anchor=(-0.01, -0.1))
    fig.subplots_adjust(bottom=0.24, top=0.97, left=0.07, right=0.99)
    save(fig, "big_overlay_ext1.png")


def big_mse_mad(seed_df):
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.3), gridspec_kw=dict(width_ratios=[2, 1.2, 3]))
    width = 0.26
    for c, (name, cfg) in enumerate(EXPERIMENTS.items()):
        part = seed_df[seed_df.dataset == name]
        horizons = list(cfg["horizons"])
        for r, measure in enumerate(("MSE", "MAD")):
            ax = axes[r, c]
            style(ax)
            ax.tick_params(labelsize=6.5)
            k = 1e5 if (name == "Exchange rate" and measure == "MSE") else 1.0
            cols = [f"{h} {measure}" for h in horizons]
            for j, label in enumerate((HYB, EXT, EXT_M)):
                rows = part[part.model == CSV_NAMES[label]][cols]
                ax.bar(np.arange(len(cols)) + (j - 1) * width, rows.mean() * k, width, yerr=rows.std() * k,
                       color=COLORS[label], edgecolor=SURFACE, linewidth=1, label=label,
                       error_kw=dict(ecolor=INK_2, elinewidth=0.8, capsize=2))
            ax.set_xticks(np.arange(len(cols)))
            ax.set_xticklabels([h.replace(" points ahead", " pts") for h in horizons], fontsize=7.5, color=INK_2)
            unit = " (x 1e-5)" if k != 1 else ""
            ax.set_title(f"{name}: {measure}{unit}", color=INK, fontsize=8.5, loc="left")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, fontsize=8.5, ncol=3, loc="lower center")
    fig.subplots_adjust(wspace=0.35, hspace=0.5, top=0.94, bottom=0.12, left=0.06, right=0.99)
    save(fig, "big_mse_mad.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    data_row()
    all_fc = {}
    for name, cfg in EXPERIMENTS.items():
        y, n = cfg["load"](), cfg["n_train"]
        p, q = cfg["ann"]
        fc = {
            HYB: HybridModel(AutoARIMA(verbose=False), p, q, seed=0).fit(y[:n]).predict_one_step(y)[n:],
            EXT: extension(cfg, 0, 0.0).fit(y[:n]).predict_one_step(y)[n:],
            EXT_M: extension(cfg, 0, 0.2).fit(y[:n]).predict_one_step(y)[n:],
        }
        all_fc[name] = (y, n, fc)
        if name == "Sunspot":
            big_overlay(y, n, fc)
            big_overlay_ext1(y, n, fc)
    big_lines(all_fc, [HYB], "big_hybrid_three.png", height=4.3)
    big_lines(all_fc, [HYB, EXT, EXT_M], "big_all_three.png", height=5.6, styled=True)
    big_mse_mad(pd.read_csv(results_path("results_extensions.csv")))
