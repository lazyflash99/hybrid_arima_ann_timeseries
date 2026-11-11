"""
Loads the 11 datasets and holds their registry (DATASETS): source, frequency, split, transformation, seasonal period
and network sizes. The first three are the paper's (Section 4.1, Table 1):

  Sunspot       : 288 annual values, 1700-1987, raw values (no transformation)
  Lynx          : 114 annual values, 1821-1934, log10 transformed
  Exchange rate : 731 weekly BP/USD values, 1980-1993, log10 transformed (see load_exchange_rate)

The other eight are listed in DATASETS.md.
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load_sunspot():
    # statsmodels provides the yearly sunspot series (1700-2008); it is cut to 1700-1987 (288 points).
    import statsmodels.api as sm
    df = sm.datasets.sunspots.load_pandas().data
    df = df[(df["YEAR"] >= 1700) & (df["YEAR"] <= 1987)]
    y = df["SUNACTIVITY"].values.astype(float)
    assert len(y) == 288, f"Sunspot: expected 288 points, got {len(y)}"
    return y


def load_lynx():
    # primary source: data/lynx.csv (one column, 114 rows, with or without a header)
    # fallback: R's 'lynx' dataset through statsmodels (requires an internet connection)
    path = os.path.join(DATA_DIR, "lynx.csv")
    if os.path.exists(path):
        y = pd.read_csv(path).iloc[:, -1].values.astype(float)
    else:
        import statsmodels.api as sm
        y = sm.datasets.get_rdataset("lynx").data["value"].values.astype(float)
    assert len(y) == 114, f"Lynx: expected 114 points, got {len(y)}"
    return np.log10(y)          # paper: logarithms to base 10


def load_exchange_rate():
    # data/exchange_rate.csv has columns: Date, USD_per_GBP, GBP_per_USD (731 weekly rows, 1980-1993):
    # Wednesday values of the Federal Reserve H.10 noon buying rate (FRED series DEXUSUK), holidays filled
    # with the previous day. The paper does not name its source; Wednesday sampling reproduces the random-walk
    # errors of Table 4 most closely among the FRED weekly series.
    # The paper text mentions natural log, but Fig. 3 & Fig. 6 (y-axis 0.14-0.22; log10(1.54) = 0.1875)
    # and Table 4 error magnitudes (order 10^-5) show that base-10 log (log10) was used in the calculations.
    path = os.path.join(DATA_DIR, "exchange_rate.csv")
    if not os.path.exists(path):
        raise FileNotFoundError("Put exchange_rate.csv in the data/ folder. See README.md.")
    df = pd.read_csv(path, parse_dates=["Date"]).sort_values("Date")
    y = df["USD_per_GBP"].values.astype(float)
    assert len(y) == 731, f"Exchange rate: expected 731 points, got {len(y)}"
    assert not np.isnan(y).any(), "Exchange rate: missing values found"
    return np.log10(y)          # paper calculations/plots use log10


# ---------------------------------------------------------------------------- additional data sets
def _csv(name):
    return os.path.join(DATA_DIR, name)


def load_airpassengers():
    return np.log10(pd.read_csv(_csv("airpassengers.csv"))["value"].values.astype(float))


def load_nile():
    import statsmodels.api as sm
    return sm.datasets.nile.load_pandas().data["volume"].values.astype(float)


def load_daily_min_temp():
    # last three years only (1988-1990), to limit the cost of repeated refits
    return pd.read_csv(_csv("daily_min_temperatures.csv"))["Temp"].values.astype(float)[-1095:]


def load_fx_jpy():
    d = pd.read_csv(_csv("fred_dexjpus.csv"), parse_dates=["observation_date"], na_values=".")
    s = d.set_index("observation_date")["DEXJPUS"].ffill().resample("W-WED").last()
    s = s[(s.index >= "1980-01-01") & (s.index <= "1993-12-31")]
    assert len(s) == 731 and s.notna().all()
    return np.log10(s.values.astype(float))


def load_sunspot_sqrt():
    return np.sqrt(load_sunspot())


def _ar2_noise(n, seed, sigma=1.0):
    rng = np.random.RandomState(seed)
    e, x = rng.normal(0, sigma, n + 100), np.zeros(n + 100)
    for t in range(2, n + 100):
        x[t] = 0.6 * x[t - 1] - 0.3 * x[t - 2] + e[t]
    return x[100:]


def synthetic_additive(n=400, seed=123):
    """y = L + N with L an AR(2) and N = 3 sin(y_{t-1}), a known nonlinear part. Returns (y, N)."""
    rng = np.random.RandomState(seed)
    y, N = np.zeros(n + 100), np.zeros(n + 100)
    for t in range(2, n + 100):
        N[t] = 3.0 * np.sin(y[t - 1])
        y[t] = 0.6 * y[t - 1] - 0.3 * y[t - 2] + N[t] + rng.normal(0, 0.5)
    return y[100:], N[100:]


def synthetic_multiplicative(n=400, seed=124):
    """y = L * N with L a positive AR level and N = 1 + 0.4 sin(2 L_{t-1}), plus small noise. Returns (y, N)."""
    L = 10.0 + _ar2_noise(n + 1, seed)
    N = 1.0 + 0.4 * np.sin(2.0 * L[:-1])
    y = L[1:] * N * np.exp(np.random.RandomState(seed + 1).normal(0, 0.02, n))
    return y, N


def mackey_glass(n=500, tau=17, seed=125, delta=1.0, sample=6):
    """Chaotic Mackey-Glass series (Euler steps, sampled every `sample` steps). The whole series is nonlinear."""
    rng = np.random.RandomState(seed)
    total = (n + 300) * sample
    x = np.zeros(total + tau + 1)
    x[:tau + 1] = 1.2 + 0.1 * rng.rand(tau + 1)
    for t in range(tau, total + tau):
        x[t + 1] = x[t] + delta * (0.2 * x[t - tau] / (1.0 + x[t - tau] ** 10) - 0.1 * x[t])
    return x[tau + 1:][::sample][300:300 + n]


def load_syn_additive():
    return synthetic_additive()[0]


def load_syn_multiplicative():
    return synthetic_multiplicative()[0]


def _entry(load, source, frequency, n_train, n_test, transform, period, ann, ext, horizons, subset=False):
    return dict(load=load, source=source, frequency=frequency, n_train=n_train, n_test=n_test,
                transform=transform, period=period, ann=ann,
                ext_kwargs=dict(n_series_lags=ext[0], n_residual_lags=ext[1], n_hidden=ext[2], weight_decay=ext[3]),
                horizons=horizons, arima_subset=subset)


# ann = (inputs, hidden nodes); ext = (past values, past errors, hidden nodes, weight decay).
# The first three use the sizes reported in the paper; the others were set before any test result was seen.
DATASETS = {
    "Sunspot": _entry(load_sunspot, "statsmodels sunspots (yearly, cut to 1700-1987)", "yearly", 221, 67, "none", 11,
                      (4, 4), (4, 2, 4, 1e-5), {"35 points ahead": 35, "67 points ahead": 67}, subset=True),
    "Lynx": _entry(load_lynx, "R lynx data set (data/lynx.csv)", "yearly", 100, 14, "log10", 10,
                   (7, 5), (7, 3, 5, 5e-4), {"14 points ahead": 14}, subset=True),
    "Exchange rate": _entry(load_exchange_rate, "FRED DEXUSUK, Wednesdays", "weekly", 679, 52, "log10", 1,
                            (7, 6), (7, 3, 6, 1e-5), {"1 month": 4, "6 months": 26, "12 months": 52}, subset=True),
    "AirPassengers": _entry(load_airpassengers, "R AirPassengers (data/airpassengers.csv)", "monthly", 120, 24, "log10", 12,
                            (12, 4), (12, 3, 4, 1e-5), {"12 points ahead": 12, "24 points ahead": 24}),
    "Nile": _entry(load_nile, "statsmodels nile", "yearly", 80, 20, "none", 1,
                   (4, 3), (4, 2, 3, 1e-5), {"10 points ahead": 10, "20 points ahead": 20}),
    "Daily min temp": _entry(load_daily_min_temp, "Melbourne daily minimum temperature, last 1095 days "
                             "(github.com/jbrownlee/Datasets)", "daily", 915, 180, "none", 365,
                             (7, 4), (7, 3, 4, 1e-5), {"90 points ahead": 90, "180 points ahead": 180}),
    "FX JPY": _entry(load_fx_jpy, "FRED DEXJPUS, Wednesdays", "weekly", 679, 52, "log10", 1,
                     (7, 6), (7, 3, 6, 1e-5), {"1 month": 4, "6 months": 26, "12 months": 52}),
    "Sunspot sqrt": _entry(load_sunspot_sqrt, "statsmodels sunspots, square root", "yearly", 221, 67, "sqrt", 11,
                           (4, 4), (4, 2, 4, 1e-5), {"35 points ahead": 35, "67 points ahead": 67}),
    "Synthetic additive": _entry(load_syn_additive, "generated: AR(2) + 3 sin(y_{t-1})", "synthetic", 320, 80, "none", 1,
                                 (4, 4), (4, 2, 4, 1e-5), {"40 points ahead": 40, "80 points ahead": 80}),
    "Synthetic multiplicative": _entry(load_syn_multiplicative, "generated: L_t x (1 + 0.4 sin(2 L_{t-1}))", "synthetic",
                                       320, 80, "none", 1, (4, 4), (4, 2, 4, 1e-5),
                                       {"40 points ahead": 40, "80 points ahead": 80}),
    "Mackey-Glass": _entry(mackey_glass, "generated: Mackey-Glass, tau = 17", "synthetic", 400, 100, "none", 1,
                           (6, 4), (6, 2, 4, 1e-5), {"50 points ahead": 50, "100 points ahead": 100}),
}


# where each series is stored
LOCATION = {
    "Sunspot": "bundled with statsmodels (`sm.datasets.sunspots`)",
    "Lynx": "`data/lynx.csv`",
    "Exchange rate": "`data/exchange_rate.csv`",
    "AirPassengers": "`data/airpassengers.csv`",
    "Nile": "bundled with statsmodels (`sm.datasets.nile`)",
    "Daily min temp": "`data/daily_min_temperatures.csv`",
    "FX JPY": "`data/fred_dexjpus.csv`",
    "Sunspot sqrt": "bundled with statsmodels (square root taken in `data_loader.py`)",
    "Synthetic additive": "generated in `data_loader.py` (seed 123)",
    "Synthetic multiplicative": "generated in `data_loader.py` (seed 124)",
    "Mackey-Glass": "generated in `data_loader.py` (seed 125)",
}


def write_datasets_md(path):
    lines = ["# Datasets", "",
             "All 11 series load offline. Five are CSV files in `data/`, three are bundled with the statsmodels package, "
             "and three are generated by `data_loader.py` with a fixed seed. `data_loader.DATASETS` holds the registry below.", "",
             "| Name | Stored in | Source | Frequency | Length | Train / test | Transformation | Season period |",
             "|---|---|---|---|---|---|---|---|"]
    for name, d in DATASETS.items():
        lines.append(f"| {name} | {LOCATION[name]} | {d['source']} | {d['frequency']} | {d['n_train'] + d['n_test']} | "
                     f"{d['n_train']} / {d['n_test']} | {d['transform']} | {d['period']} |")
    lines += ["", "Generated by `data_loader.write_datasets_md`."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
