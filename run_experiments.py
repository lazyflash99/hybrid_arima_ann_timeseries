"""
Runs ARIMA, ANN and Hybrid on the three data sets and prints MSE / MAD tables
in the same layout as Tables 2, 3 and 4 of the paper.   Usage:  python run_experiments.py

Extension (not in the paper): Extension hybrid with a scale margin. One network combines the past values,
the ARIMA forecast and the past ARIMA errors (no fixed L + N sum), and the training range is mapped to
[0.2, 0.8] instead of [0, 1], which keeps inputs beyond the training range (the 1957-59 sunspot peak) away from
the flat ends of the logistic hidden units. See extension_model.py.
"""
import numpy as np
from config import figure_path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_loader import load_sunspot, load_lynx, load_exchange_rate
from arima_model import ARIMA, AutoARIMA
from ann_model import NeuralNetwork

from hybrid_model import HybridModel
from extension_model import ExtensionHybrid

# Table 1 sizes, ANN structures (inputs x hidden x 1), forecast horizons (number of test points)
# Linear model: "auto" = Box-Jenkins selection on the training data (most parsimonious adequate ARIMA),
#               "paper" = the models reported in Section 4.2 (the `linear` entry of each experiment).
ARIMA_SELECTION = "auto"
EXT_SCALE_MARGIN = 0.2          # extension: training range mapped to [0.2, 0.8]


def make_linear(cfg):
    return AutoARIMA() if ARIMA_SELECTION == "auto" else cfg["linear"]()


EXPERIMENTS = {
    "Sunspot": dict(
        load=load_sunspot, n_train=221, n_test=67,
        linear=lambda: ARIMA(p=[1, 2, 9], d=0, q=0),                    # subset AR of order 9
        ann=(4, 4),
        ann_kwargs=dict(seed=0),
        hybrid_ann_kwargs=dict(seed=0),
        ext_kwargs=dict(n_series_lags=4, n_residual_lags=2, n_hidden=4, weight_decay=1e-5),
        horizons={"35 points ahead": 35, "67 points ahead": 67}
    ),
    "Lynx": dict(
        load=load_lynx, n_train=100, n_test=14,
        # Subset AR model of order 12 from Subba Rao & Gabr (1984) / Campbell & Walker (1977)
        linear=lambda: ARIMA(p=[1, 2, 4, 6, 9, 10, 11, 12], d=0, q=0),
        ann=(7, 5),
        ann_kwargs=dict(seed=0),
        hybrid_ann_kwargs=dict(seed=0),
        ext_kwargs=dict(n_series_lags=7, n_residual_lags=3, n_hidden=5, weight_decay=5e-4),
        horizons={"14 points ahead": 14}
    ),
    "Exchange rate": dict(
        load=load_exchange_rate, n_train=679, n_test=52,
        linear=lambda: ARIMA(p=0, d=1, q=0, constant=False),       # random walk y_t = y_{t-1} + e_t
        ann=(7, 6),
        ann_kwargs=dict(seed=0),
        hybrid_ann_kwargs=dict(seed=0),
        ext_kwargs=dict(n_series_lags=7, n_residual_lags=3, n_hidden=6, weight_decay=1e-5),
        # 1 month = 4 weeks, 6 months = 26 weeks (or 24 weeks), 12 months = 52 weeks
        horizons={"1 month": 4, "6 months": 26, "12 months": 52}
    ),
}


def mse(a, f): return np.mean((a - f) ** 2)
def mad(a, f): return np.mean(np.abs(a - f))


def improvement(error_other, error_hybrid):
    """Percentage improvement as computed in the paper, with the hybrid model's error as the denominator.
       (error_other - error_hybrid) / error_hybrid * 100
       e.g. sunspot 35 pts MSE: (216.965 - 186.827) / 186.827 = 16.13 %  (paper's text)."""
    return (error_other - error_hybrid) / error_hybrid * 100


def run(name, cfg):
    print(f"\n===== {name} =====")
    y = cfg["load"]()
    n_train, n_test = cfg["n_train"], cfg["n_test"]
    assert len(y) == n_train + n_test, "data length does not match Table 1"
    y_train, y_test = y[:n_train], y[n_train:]
    p, q = cfg["ann"]
    ann_kw = cfg.get("ann_kwargs", {})
    hyb_ann_kw = cfg.get("hybrid_ann_kwargs", ann_kw)

    # 1) ARIMA alone
    print(" ARIMA")
    arima = make_linear(cfg).fit(y_train)
    f_arima = arima.predict_one_step(y)[n_train:]

    # 2) ANN alone
    print(" ANN")
    ann = NeuralNetwork(p, q, **ann_kw).fit(y_train)
    f_ann = ann.predict_one_step(y)[n_train:]
    print(f"   train MSE {ann.train_mse:.6g}, max |weight| {ann.max_abs_weight:.2f}, BFGS iterations {ann.n_iter}")

    # 3) Hybrid (ARIMA + ANN on ARIMA residuals)
    print(" Hybrid")
    hybrid = HybridModel(make_linear(cfg), n_inputs=p, n_hidden=q, **hyb_ann_kw).fit(y_train)
    f_hyb = hybrid.predict_one_step(y)[n_train:]
    print(f"   residual-ANN train MSE {hybrid.ann.train_mse:.6g}, max |weight| {hybrid.ann.max_abs_weight:.2f}, BFGS iterations {hybrid.ann.n_iter}")

    # 4) Extension with scale margin
    print(" Extension + margin")
    ext = ExtensionHybrid(make_linear(cfg), scale_margin=EXT_SCALE_MARGIN, n_restarts=20, seed=0,
                             **cfg["ext_kwargs"]).fit(y_train)
    f_ext = ext.predict_one_step(y)[n_train:]

    forecasts = {"ARIMA": f_arima, "ANN": f_ann, "Hybrid": f_hyb, "EXT+margin": f_ext}

    # results table
    scale_mse = 1e5 if name == "Exchange rate" else 1.0
    note_str = " (Note: MSE multiplied by 10^5 as in Table 4)" if name == "Exchange rate" else ""
    header = "".join(f"{h:>34}" for h in cfg["horizons"])
    print(f"\n{'':10}{header}{note_str}")
    print(f"{'':10}" + "".join(f"{'MSE':>17}{'MAD':>17}" for _ in cfg["horizons"]))
    for model, f in forecasts.items():
        row = f"{model:10}"
        for h in cfg["horizons"].values():
            m_val = mse(y_test[:h], f[:h]) * scale_mse
            row += f"{m_val:17.6g}{mad(y_test[:h], f[:h]):17.6g}"
        print(row)

    # percentage improvement of the hybrid over ARIMA and over ANN (paper's formula)
    print("\n Hybrid improvement (%), denominator = hybrid error")
    for h_name, h in cfg["horizons"].items():
        for metric_name, metric in (("MSE", mse), ("MAD", mad)):
            e_h = metric(y_test[:h], f_hyb[:h])
            over_arima = improvement(metric(y_test[:h], f_arima[:h]), e_h)
            over_ann = improvement(metric(y_test[:h], f_ann[:h]), e_h)
            print(f"   {h_name:16} {metric_name}: over ARIMA {over_arima:7.2f}%   over ANN {over_ann:7.2f}%")

    # extension vs the paper's models (same formula, denominator = extension's error)
    print("\n Extension (Extension + margin) improvement (%), denominator = extension error")
    for h_name, h in cfg["horizons"].items():
        for metric_name, metric in (("MSE", mse), ("MAD", mad)):
            e_ext = metric(y_test[:h], f_ext[:h])
            vals = "   ".join(f"over {m} {improvement(metric(y_test[:h], forecasts[m][:h]), e_ext):7.2f}%"
                             for m in ("ARIMA", "ANN", "Hybrid"))
            print(f"   {h_name:16} {metric_name}: {vals}")

    # actual against forecast, in the layout of the paper's Figs. 4-6
    fig, axes = plt.subplots(4, 1, figsize=(7, 10.5))
    letters = ["(a)", "(b)", "(c)", "(d)"]
    x_idx = np.arange(1, len(y_test) + 1)
    
    # axis ranges as in the paper
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

    for i, (ax, (model, f)) in enumerate(zip(axes, forecasts.items())):
        ax.plot(x_idx, y_test, "k-", linewidth=1.2, label="actual")
        ax.plot(x_idx, f, "k:", linewidth=1.4, dashes=(1, 2), label="prediction")
        title = "Extension (Extension + margin)" if model == "EXT+margin" else model
        ax.set_title(f"{letters[i]} {title} prediction of {name.lower()}", fontsize=10, loc="left", pad=4)
        ax.set_xlim(1, len(y_test))
        if "ylim" in spec:
            ax.set_ylim(spec["ylim"])
        if "yticks" in spec:
            ax.set_yticks(spec["yticks"])
        if "xticks" in spec:
            ax.set_xticks(spec["xticks"])
        ax.tick_params(direction="in", top=True, right=True, labelsize=8)
        for label in ax.get_xticklabels():
            label.set_fontsize(7)

    # one legend below the last panel
    axes[-1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=True, fontsize=9)
    fig.subplots_adjust(hspace=0.45, bottom=0.09, top=0.97, left=0.12, right=0.95)
    fig.savefig(figure_path("paper", f"results_{name.replace(' ', '_').lower()}.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for name, cfg in EXPERIMENTS.items():
        run(name, cfg)
