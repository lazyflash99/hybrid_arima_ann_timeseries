"""
Slide figure: accumulated squared forecast error over the test period for the hybrid, Extension 1 and
Extension 2 (mean over 10 random starts). A steeper line means larger errors; the final height divided by
the number of test points is the MSE.   Usage:  python make_error_figure.py
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor
import numpy as np

from config import figure_path

SEEDS = range(10)
MODELS = ["Hybrid (Zhang)", "Extension 1", "Extension 1 Improved"]
STYLE = {"Hybrid (Zhang)": dict(color="#1f4e9c", linestyle="-", linewidth=2.2),
         "Extension 1": dict(color="#d95f02", linestyle=(0, (5, 2)), linewidth=2.2),
         "Extension 1 Improved": dict(color="#0f8a5f", linestyle=(0, (1.5, 1.5)), linewidth=2.6)}
START_YEAR = {"Sunspot": 1700, "Lynx": 1821}


def squared_errors(job):
    from arima_model import AutoARIMA
    from hybrid_model import HybridModel
    from extension_model import ExtensionHybrid
    from run_experiments import EXPERIMENTS
    name, model, seed = job
    cfg = EXPERIMENTS[name]
    y, n = cfg["load"](), cfg["n_train"]
    p, q = cfg["ann"]
    ext = dict(cfg["ext_kwargs"], n_restarts=20, seed=seed)
    m = {"Hybrid (Zhang)": lambda: HybridModel(AutoARIMA(verbose=False), p, q, seed=seed),
         "Extension 1": lambda: ExtensionHybrid(AutoARIMA(verbose=False), **ext),
         "Extension 1 Improved": lambda: ExtensionHybrid(AutoARIMA(verbose=False), scale_margin=0.2, **ext)}[model]()
    f = m.fit(y[:n]).predict_one_step(y)[n:]
    return name, model, seed, (y[n:] - f) ** 2


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from run_experiments import EXPERIMENTS

    jobs = [(name, model, s) for name in EXPERIMENTS for model in MODELS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(squared_errors, jobs))
    se = {}
    for name, model, seed, e in results:
        se.setdefault((name, model), []).append(e)

    fig, axes = plt.subplots(1, 3, figsize=(6.8, 3.0))
    panels = []
    for ax, name in zip(axes, EXPERIMENTS):
        y, n = EXPERIMENTS[name]["load"](), EXPERIMENTS[name]["n_train"]
        T = len(y) - n
        x = (np.arange(n, len(y)) + START_YEAR[name]) if name in START_YEAR else np.arange(1, T + 1)
        k = 1e5 if name == "Exchange rate" else 1.0
        if name == "Sunspot":
            ax.axvspan(1956.5, 1959.5, color="#c0392b", alpha=0.12, linewidth=0)
            ax.text(1958, 0.97, "record\nyears", transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=6.5, color="#c0392b")
        ends = []
        for model in MODELS:
            cum = np.cumsum(np.mean(se[(name, model)], axis=0)) * k
            ax.plot(x, cum, label=model, **STYLE[model])
            ends.append((cum[-1], model))
        panels.append((ax, x[-1], T, ends))
        title = "Exchange (x 1e-5)" if k != 1 else name
        ax.set_title(title, fontsize=9, loc="left", color="#0b0b0b", pad=8)
        ax.set_xlim(x[0], x[-1])
        ax.tick_params(labelsize=7, colors="#52514e")
        ax.grid(axis="y", color="#e4e3df", linewidth=0.7)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("accumulated squared error", fontsize=7.5, color="#52514e")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               handlelength=3)
    fig.subplots_adjust(wspace=0.55, top=0.9, bottom=0.24, left=0.1, right=0.92)

    # direct labels at the right end (final MSE = total / T), spaced at least 9 px apart vertically
    fig.canvas.draw()
    for ax, x_end, T, ends in panels:
        ends.sort()
        last_px = -np.inf
        for v, model in ends:
            px = max(ax.transData.transform((x_end, v))[1], last_px + 9 * fig.dpi / 72)
            last_px = px
            y_label = ax.transData.inverted().transform((0, px))[1]
            ax.annotate(f"{v / T:.3g}", xy=(x_end, y_label), xytext=(4, 0), textcoords="offset points",
                        fontsize=7, color=STYLE[model]["color"], fontweight="bold", va="center",
                        annotation_clip=False)
    path = figure_path("slides", "big_error_accumulation.png")
    fig.savefig(path, dpi=200, facecolor="white")
    print("wrote", path)
    for name in EXPERIMENTS:
        T = len(EXPERIMENTS[name]["load"]()) - EXPERIMENTS[name]["n_train"]
        print(name, {m: round(float(np.mean(np.sum(se[(name, m)], axis=1)) / T), 6) for m in MODELS})


if __name__ == "__main__":
    main()
