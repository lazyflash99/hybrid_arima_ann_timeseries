
import json
import os
import sys
import time

import numpy as np
from joblib import Memory, Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import arima_model as am  # noqa: E402
from config import MODES, results_path  # noqa: E402
from data_loader import DATASETS, _ar2_noise  # noqa: E402
from extension_model import ExtensionHybrid  # noqa: E402
from extensions import MultiplicativeHybrid  # noqa: E402
from hybrid_model import HybridModel  # noqa: E402
from validation import refit_origins  # noqa: E402

OUT = os.path.join(HERE, "data")
memory = Memory(os.path.join(HERE, ".cache"), verbose=0)

SETTINGS = {
    "quick": dict(strengths=[0.0, 1.0, 2.0, 3.0], noises=[0.25, 0.5, 1.0, 2.0], seeds=[0, 1, 2], restarts=5,
                  lab_strengths=[0.0, 0.15, 0.3, 0.45], margin_seeds=[0, 1, 2]),
    "full": dict(strengths=[0.0, 0.6, 1.2, 1.8, 2.4, 3.0], noises=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0], seeds=[0, 1, 2],
                 restarts=10, lab_strengths=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5], margin_seeds=list(range(10))),
}
N_SERIES, N_TRAIN = 300, 240          # synthetic series: 240 training points, 60 test points
LAGS, HIDDEN, ERRORS = 4, 4, 2       # network sizes, fixed before any result was seen


def write(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, allow_nan=False)          # fails on NaN or inf
    print("wrote", name)


def r6(a):
    return [round(float(v), 6) for v in np.asarray(a, float)]


# ------------------------------------------------------------------ synthetic series
def additive_ar_sin(strength, noise, seed, n=N_SERIES):
    """y_t = 0.6 y_{t-1} - 0.3 y_{t-2} + strength * sin(y_{t-1}) + noise * e_t (the Synthetic additive dataset uses 3 and 0.5)."""
    rng = np.random.RandomState(1000 + seed)
    y = np.zeros(n + 100)
    for t in range(2, n + 100):
        y[t] = 0.6 * y[t - 1] - 0.3 * y[t - 2] + strength * np.sin(y[t - 1]) + noise * rng.normal()
    return y[100:]


def level_series(form, strength, seed, n=N_SERIES):
    """Persistent positive level L_t = 10 exp(0.3 x_t), x an AR(2), and a nonlinear term driven by the previous
    observation, g_t = sin(y_{t-1} - 10):
      multiplicative: y_t = L_t (1 + s g_t) + v_t      additive: y_t = L_t + 10 s g_t + v_t,      v_t ~ N(0, 0.2^2).
    Both forms have the same average amplitude; in the multiplicative form it grows with the level."""
    L = 10.0 * np.exp(0.3 * _ar2_noise(n, 2000 + seed))
    v = np.random.RandomState(3000 + seed).normal(0, 0.2, n)
    y = np.zeros(n)
    y[0] = L[0]
    for t in range(1, n):
        g = np.sin(y[t - 1] - 10.0)
        y[t] = (L[t] * (1.0 + strength * g) if form == "multiplicative" else L[t] + 10.0 * strength * g) + v[t]
    return y


# ------------------------------------------------------------------ one fit
def model(name, restarts):
    arima = lambda: am.AutoARIMA(verbose=False, subset=False)           # noqa: E731
    if name == "ARIMA":
        return arima()
    if name == "Zhang":
        return HybridModel(arima(), LAGS, HIDDEN, seed=0)
    if name == "Extension":
        return ExtensionHybrid(arima(), n_series_lags=LAGS, n_residual_lags=ERRORS, n_hidden=HIDDEN, n_restarts=restarts, seed=0)
    if name == "E2":
        return MultiplicativeHybrid(LAGS, HIDDEN, seed=0)
    raise ValueError(name)


@memory.cache
def test_mse(kind, args, model_name, restarts):
    y = additive_ar_sin(*args) if kind == "break_even" else level_series(*args)
    f = model(model_name, restarts).fit(y[:N_TRAIN]).predict_one_step(y)[N_TRAIN:]
    if not np.isfinite(f).all():
        raise ValueError("non-finite forecast")
    return float(np.mean((y[N_TRAIN:] - f) ** 2))


@memory.cache
def arima_is_constant(form, strength, seed):
    m = am.AutoARIMA(verbose=False, subset=False).fit(level_series(form, strength, seed)[:N_TRAIN]).model
    return bool(not m.ar_lags and not m.ma_lags and m.d == 0)


@memory.cache
def margin_forecast(margin, seed):
    d = DATASETS["Sunspot"]
    am.DEFAULT_SUBSET = d["arima_subset"]
    y, n = d["load"](), d["n_train"]
    m = ExtensionHybrid(am.AutoARIMA(verbose=False), scale_margin=margin, n_restarts=20, seed=seed, **d["ext_kwargs"])
    return m.fit(y[:n]).predict_one_step(y)[n:]


def run(jobs, n_jobs=-2):
    return Parallel(n_jobs=n_jobs)(delayed(test_mse)(*j) for j in jobs)


# ------------------------------------------------------------------ the five files
def break_even(cfg):
    models = ["ARIMA", "Zhang", "Extension"]
    jobs = [("break_even", (s, nz, seed), m, cfg["restarts"]) for s in cfg["strengths"] for nz in cfg["noises"]
            for seed in cfg["seeds"] for m in models]
    mse = dict(zip(jobs, run(jobs)))
    cells = []
    for s in cfg["strengths"]:
        for nz in cfg["noises"]:
            per = {m: [mse[("break_even", (s, nz, seed), m, cfg["restarts"])] for seed in cfg["seeds"]] for m in models}
            means = {m: float(np.mean(v)) for m, v in per.items()}
            ranked = sorted(means, key=means.get)
            winner, runner = ranked[0], ranked[1]
            wins = int(sum(a < b for a, b in zip(per[winner], per[runner])))
            cells.append(dict(strength=s, noise=nz, mse_mean={m: round(v, 6) for m, v in means.items()},
                              mse_by_seed={m: r6(v) for m, v in per.items()}, winner=winner, runner_up=runner,
                              winner_beats_runner_up_in_seeds=wins, n_seeds=len(cfg["seeds"]),
                              consistent=wins == len(cfg["seeds"])))
    write("break_even.json", dict(
        description="y_t = 0.6 y_{t-1} - 0.3 y_{t-2} + strength * sin(y_{t-1}) + noise * e_t; "
                    f"{N_TRAIN} training and {N_SERIES - N_TRAIN} test points; one-step test MSE",
        strengths=cfg["strengths"], noises=cfg["noises"], seeds=cfg["seeds"], models=models,
        network=dict(lags=LAGS, hidden=HIDDEN, errors=ERRORS, extension_restarts=cfg["restarts"]), cells=cells))


def form_lab(cfg):
    models = ["ARIMA", "Zhang", "Extension", "E2"]
    forms = ["additive", "multiplicative"]
    jobs = [("form_lab", (form, s, seed), m, cfg["restarts"]) for form in forms for s in cfg["lab_strengths"]
            for seed in cfg["seeds"] for m in models]
    mse = dict(zip(jobs, run(jobs)))
    rows = []
    for form in forms:
        for s in cfg["lab_strengths"]:
            per = {m: [mse[("form_lab", (form, s, seed), m, cfg["restarts"])] for seed in cfg["seeds"]] for m in models}
            # when ARIMA reduces to a constant forecast, E2's ratio input is a linear rescaling of Zhang's error input,
            # so the two give identical forecasts; record how often that happens
            constant = [arima_is_constant(form, s, seed) for seed in cfg["seeds"]]
            rows.append(dict(form=form, strength=s, mse_mean={m: round(float(np.mean(v)), 6) for m, v in per.items()},
                             mse_std={m: round(float(np.std(v)), 6) for m, v in per.items()}, mse_by_seed={m: r6(v) for m, v in per.items()},
                             arima_constant_share=round(float(np.mean(constant)), 6)))
    write("form_lab.json", dict(
        description="L_t = 10 exp(0.3 x_t) with x an AR(2); g_t = sin(y_{t-1} - 10); multiplicative: y = L (1 + s g) + noise; "
                    "additive: y = L + 10 s g + noise; noise ~ N(0, 0.2^2)",
        strengths=cfg["lab_strengths"], forms=forms, seeds=cfg["seeds"], models=models, rows=rows))


def rolling_animation():
    store = np.load(results_path("forecasts_rolling.npz"))
    out = {}
    for name in ("Lynx", "Nile"):
        d = DATASETS[name]
        y, n = d["load"](), d["n_train"]
        origins = refit_origins(n, len(y), MODES["full"]["n_refits"])
        refit_at = [max(o for o in origins if o <= t) for t in range(n, len(y))]
        fc = {}
        for m in ("ARIMA", "Zhang", "Extension"):
            arr = np.array([store[k] for k in store.files if k.startswith(f"{name}|{m}|")])
            fc[m] = r6(arr.mean(0))
        out[name] = dict(series=r6(y), n_train=n, test_index=list(range(n, len(y))), refit_origins=origins,
                         fitted_through=refit_at, forecasts=fc, n_seeds={m: int(sum(k.startswith(f"{name}|{m}|") for k in store.files))
                                                                        for m in fc})
    write("rolling_animation.json", dict(
        description="rolling protocol: one-step forecast of every test point, made with the model refitted at the last "
                    "refit origin before it (expanding window); forecasts are averaged over seeds",
        datasets=out))


def seed_fan():
    store = np.load(results_path("forecasts_fixed.npz"))
    out = {}
    for name in ("Sunspot", "Lynx", "Synthetic additive"):
        d = DATASETS[name]
        y, n = d["load"](), d["n_train"]
        out[name] = dict(actual_test=r6(y[n:]), n_train=n, forecasts={
            m: {k.split("|")[2]: r6(store[k]) for k in store.files if k.startswith(f"{name}|{m}|")} for m in ("Zhang", "Extension")})
    write("seed_fan.json", dict(description="fixed split: test forecasts of every seed (0-9)", datasets=out))


def scale_margin(cfg):
    margins = [0.0, 0.1, 0.2, 0.3]
    d = DATASETS["Sunspot"]
    y, n = d["load"](), d["n_train"]
    jobs = [(m, s) for m in margins for s in cfg["margin_seeds"]]
    res = dict(zip(jobs, Parallel(n_jobs=-2)(delayed(margin_forecast)(m, s) for m, s in jobs)))
    a = y[n:]
    fc = {}
    for m in margins:
        arr = np.array([res[(m, s)] for s in cfg["margin_seeds"]])
        f = arr.mean(0)
        above = a > y[:n].max()
        fc[f"{m:g}"] = dict(forecast=r6(f), mse=round(float(np.mean((a - f) ** 2)), 6),
                          mse_record_points=round(float(np.mean((a[above] - f[above]) ** 2)), 6),
                          max_forecast=round(float(f.max()), 6))
    write("scale_margin.json", dict(
        description="Sunspot, fixed split: Extension test forecasts (mean over seeds) for each scale margin",
        years=list(range(1700 + n, 1700 + len(y))), actual_test=r6(a), train_min=round(float(y[:n].min()), 6),
        train_max=round(float(y[:n].max()), 6), margins=margins, seeds=cfg["margin_seeds"],
        record_points=[int(i) for i in np.where(a > y[:n].max())[0]], forecasts=fc))


def main(mode):
    cfg = SETTINGS[mode]
    t0 = time.time()
    for step in (lambda: break_even(cfg), lambda: form_lab(cfg), rolling_animation, seed_fan, lambda: scale_margin(cfg)):
        t = time.time()
        step()
        print(f"   {time.time() - t:.0f} s", flush=True)
    print(f"total {time.time() - t0:.0f} s ({mode})")


if __name__ == "__main__":
    main("quick" if "--quick" in sys.argv else "full")
