"""
ARIMA, ANN, Zhang's hybrid and the Extension on the three original datasets (single run, seed 0): error tables,
percentage improvements and a four-panel plot per dataset.   Usage:  python run_extension_experiments.py
"""
import numpy as np
from config import figure_path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from arima_model import AutoARIMA
from ann_model import NeuralNetwork
from hybrid_model import HybridModel
from extension_model import ExtensionHybrid


# Linear model: "auto" = Box-Jenkins selection on the training data (most parsimonious adequate ARIMA),
#               "paper" = the models reported in Section 4.2 (the `linear` entry of each experiment).
ARIMA_SELECTION = "auto"


def make_linear(cfg):
    return AutoARIMA() if ARIMA_SELECTION == "auto" else cfg["linear"]()


from run_experiments import EXPERIMENTS          # one shared definition


def mse(a, f): return np.mean((a - f) ** 2)
def mad(a, f): return np.mean(np.abs(a - f))


def pct_improvement(error_baseline, error_target):
    """
    Percentage improvement using target model error as denominator:
    (error_baseline - error_target) / error_target * 100
    (matches Zhang 2003 convention)
    """
    return (error_baseline - error_target) / error_target * 100


def run_benchmark(name, cfg):
    print(f"\n=======================================================")
    print(f" Dataset: {name}")
    print(f"=======================================================")
    y = cfg["load"]()
    n_train, n_test = cfg["n_train"], cfg["n_test"]
    assert len(y) == n_train + n_test, "Data length does not match Table 1"
    y_train, y_test = y[:n_train], y[n_train:]
    p, q = cfg["ann"]
    ann_kw = cfg.get("ann_kwargs", {})
    hyb_ann_kw = cfg.get("hybrid_ann_kwargs", ann_kw)
    ext_kw = cfg.get("ext_kwargs", {})

    print("[1/4] ARIMA")
    arima = make_linear(cfg).fit(y_train)
    f_arima = arima.predict_one_step(y)[n_train:]

    print("[2/4] ANN")
    ann = NeuralNetwork(p, q, **ann_kw).fit(y_train)
    f_ann = ann.predict_one_step(y)[n_train:]

    print("[3/4] Zhang's hybrid")
    zhang_hyb = HybridModel(make_linear(cfg), n_inputs=p, n_hidden=q, **hyb_ann_kw).fit(y_train)
    f_zhang = zhang_hyb.predict_one_step(y)[n_train:]

    print("[4/4] Extension")
    ext_hyb = ExtensionHybrid(make_linear(cfg), seed=0, **ext_kw).fit(y_train)
    f_ext = ext_hyb.predict_one_step(y)[n_train:]

    forecasts = {
        "ARIMA": f_arima,
        "ANN (Zhang)": f_ann,
        "Zhang Hybrid": f_zhang,
        "Extension": f_ext,
    }

    # error table (exchange-rate MSE multiplied by 10^5)
    scale_mse = 1e5 if name == "Exchange rate" else 1.0
    note_str = " (Note: MSE multiplied by 10^5)" if name == "Exchange rate" else ""
    header = "".join(f"{h:>34}" for h in cfg["horizons"])
    print(f"\n{'Model':18}{header}{note_str}")
    print(f"{'':18}" + "".join(f"{'MSE':>17}{'MAD':>17}" for _ in cfg["horizons"]))
    print("-" * (18 + 34 * len(cfg["horizons"])))

    for model_name, f in forecasts.items():
        row = f"{model_name:18}"
        for h in cfg["horizons"].values():
            m_val = mse(y_test[:h], f[:h]) * scale_mse
            d_val = mad(y_test[:h], f[:h])
            row += f"{m_val:17.5f}{d_val:17.5f}"
        print(row)

    # percentage improvement of the Extension, with the Extension's error as denominator
    print(f"\n--- Extension Improvements (%) [Denominator = Extension error] ---")
    for h_name, h in cfg["horizons"].items():
        for metric_name, metric_fn in (("MSE", mse), ("MAD", mad)):
            e_ext = metric_fn(y_test[:h], f_ext[:h])
            e_arima = metric_fn(y_test[:h], f_arima[:h])
            e_ann = metric_fn(y_test[:h], f_ann[:h])
            e_zhang = metric_fn(y_test[:h], f_zhang[:h])

            imp_arima = pct_improvement(e_arima, e_ext)
            imp_ann = pct_improvement(e_ann, e_ext)
            imp_zhang = pct_improvement(e_zhang, e_ext)

            print(f"  {h_name:16} {metric_name:3}: "
                  f"vs ARIMA: {imp_arima:+7.2f}% | "
                  f"vs ANN: {imp_ann:+7.2f}% | "
                  f"vs Zhang Hybrid: {imp_zhang:+7.2f}%")

    # actual against forecast, one panel per model
    fig, axes = plt.subplots(4, 1, figsize=(8, 10))
    letters = ["(a)", "(b)", "(c)", "(d)"]
    x_idx = np.arange(1, len(y_test) + 1)

    plot_specs = {
        "Sunspot": {
            "ylim": (0, 250),
            "yticks": [0, 50, 100, 150, 200, 250],
            "xticks": [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58, 61, 64, 67]
        },
        "Lynx": {
            "ylim": (0.0, 4.0),
            "yticks": [0.0, 1.0, 2.0, 3.0, 4.0],
            "xticks": list(range(1, 15))
        },
        "Exchange rate": {
            "ylim": (0.10, 0.22),
            "yticks": [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22],
            "xticks": [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52]
        }
    }
    spec = plot_specs.get(name, {})

    for i, (ax, (m_label, f_pred)) in enumerate(zip(axes, forecasts.items())):
        ax.plot(x_idx, y_test, "k-", linewidth=1.2, label="actual")
        ax.plot(x_idx, f_pred, "r--" if "Extension" in m_label else "k:",
                linewidth=1.4, dashes=(2, 2) if "Extension" in m_label else (1, 2),
                label="prediction")
        ax.set_title(f"{letters[i]} {m_label} prediction of {name.lower()}", fontsize=10, loc="left", pad=4)
        ax.set_xlim(1, len(y_test))
        if "ylim" in spec: ax.set_ylim(spec["ylim"])
        if "yticks" in spec: ax.set_yticks(spec["yticks"])
        if "xticks" in spec: ax.set_xticks(spec["xticks"])
        ax.tick_params(direction="in", top=True, right=True, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_fontsize(7)

    axes[-1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=True, fontsize=9)
    fig.subplots_adjust(hspace=0.55, bottom=0.10, top=0.96, left=0.12, right=0.95)
    plot_filename = figure_path("paper", f"results_extension_{name.replace(' ', '_').lower()}.png")
    fig.savefig(plot_filename, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {plot_filename}")


if __name__ == "__main__":
    for name, cfg in EXPERIMENTS.items():
        run_benchmark(name, cfg)
