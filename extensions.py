"""
Extensions of Zhang (2003) that relax its central assumption y_t = L_t + N_t, where the ARIMA forecast and
the ANN residual forecast are simply added (Eq. 8).

  E1  SkipCombination       y_hat = L_hat + f(y lags, L_hat, e lags)         nonlinear combination, linear part kept
  E2  MultiplicativeHybrid  y_hat = L_hat * N_hat,  N = y / L                  (Wang et al., 2013)
  E3  DecompositionHybrid   y = A + D, A = causal Haar wavelet smooth, ARIMA on A, ANN on D
                                                                               (Khandelwal et al., 2015; Babu & Reddy, 2014)
  E4  GatedHybrid           y_hat = L_hat + G(state) * N_hat,  G in (0, 1)    state-dependent weight on the ANN part
  E5  JointARNN             y_hat = AR(y lags) + f(y lags), both trained together (no separate stages)
  E6  WeightedCombination   y_hat = c + a*L_hat + b*N_hat + g*ANN_hat          weights estimated instead of fixed 1 + 1
  E7  ValidatedCombination  as E6, but the weights come from out-of-sample forecasts on the last part of the
                            training data (the test data are never used)

Every network is the Eq. 2 network trained exactly like ann_model.NeuralNetwork (plain MSE, BFGS with the
GRG2 termination test, one run from U(-0.5, 0.5) weights), and every linear part is AutoARIMA, so the only
difference from Zhang's hybrid is how the linear and nonlinear parts are combined.
All models: fit(y_train) on the training data only, predict_one_step(y_all) with actual past values.
"""
import numpy as np
from ann_model import NeuralNetwork, grg2_minimize, sigmoid
from arima_model import AutoARIMA
from hybrid_model import HybridModel


def _rows(columns, start, end):
    """Feature matrix with one row per t in [start, end): columns is a list of (series, lag)."""
    t = np.arange(start, end)
    return np.column_stack([s[t - lag] for s, lag in columns]), t


class _Net:
    """Eq. 2 network on an arbitrary feature matrix, trained like NeuralNetwork."""

    def __init__(self, n_hidden, seed):
        self.q, self.seed = n_hidden, seed

    def fit(self, X, y):
        self.nn = NeuralNetwork(X.shape[1], self.q, seed=self.seed)
        self.w = self.nn._train(X, y)
        return self

    def predict(self, X):
        return self.nn._forward(self.w, X)[1]


class _Scaler:
    def __init__(self, y_train):
        self.lo, self.rng = np.min(y_train), np.ptp(y_train)

    def level(self, v):
        return (v - self.lo) / self.rng

    def diff(self, v):
        return v / self.rng


# ------------------------------------------------------------------------------------------- E1
class SkipCombination:
    """Network sees the series lags, the ARIMA forecast and the residual lags (as in Extension),
    but its output is a correction added to L_hat, so the ARIMA forecast reaches the output unchanged."""

    def __init__(self, n_series_lags, n_residual_lags, n_hidden, seed=0):
        self.p, self.r, self.q, self.seed = n_series_lags, n_residual_lags, n_hidden, seed

    def _columns(self, y, L):
        s, e = self.sc.level(y), self.sc.diff(y - L)
        return [(s, k) for k in range(1, self.p + 1)] + [(self.sc.level(L), 0)] + \
               [(e, k) for k in range(1, self.r + 1)], e

    def fit(self, y_train):
        self.lin = AutoARIMA(verbose=False).fit(y_train)
        self.sc = _Scaler(y_train)
        self.start = max(self.p, self.lin.max_lag + self.r)
        cols, e = self._columns(y_train, self.lin.predict_one_step(y_train))
        X, t = _rows(cols, self.start, len(y_train))
        self.net = _Net(self.q, self.seed).fit(X, e[t])
        return self

    def predict_one_step(self, y):
        L = self.lin.predict_one_step(y)
        cols, _ = self._columns(y, L)
        X, t = _rows(cols, self.start, len(y))
        out = np.full(len(y), np.nan)
        out[t] = L[t] + self.net.predict(X) * self.sc.rng
        return out


# ------------------------------------------------------------------------------------------- E2
class MultiplicativeHybrid:
    """y_t = L_t * N_t. ARIMA gives L_hat, the network forecasts the ratio N_t = y_t / L_hat_t from its past
    values. The series is shifted so its training minimum is at least 1 (ratios are undefined near 0)."""

    def __init__(self, n_inputs, n_hidden, seed=0):
        self.p, self.q, self.seed = n_inputs, n_hidden, seed

    def fit(self, y_train):
        self.shift = max(0.0, 1.0 - np.min(y_train))
        z = y_train + self.shift
        self.lin = AutoARIMA(verbose=False).fit(z)
        ratio = z / self.lin.predict_one_step(z)
        self.first = self.lin.max_lag
        self.ann = NeuralNetwork(self.p, self.q, seed=self.seed).fit(ratio[self.first:])
        return self

    def predict_one_step(self, y):
        z = y + self.shift
        L = self.lin.predict_one_step(z)
        ratio_hat = np.full(len(y), np.nan)
        ratio_hat[self.first:] = self.ann.predict_one_step((z / L)[self.first:])
        return L * ratio_hat - self.shift


# ------------------------------------------------------------------------------------------- E3
class DecompositionHybrid:
    """Causal Haar a-trous wavelet with J levels: the smooth part A_t (mean of the last 2^J values) goes to
    ARIMA, the detail part D_t = y_t - A_t to the ANN; the forecasts are recombined. The split is by
    frequency, not by an assumed linear + nonlinear structure. Uses only past values (no look-ahead).
    The smooth part needs many ARMA terms, so its ARIMA is selected among full-order models only."""

    def __init__(self, n_inputs, n_hidden, levels=2, seed=0):
        self.p, self.q, self.levels, self.seed = n_inputs, n_hidden, levels, seed
        self.w = 2 ** levels

    def _split(self, y):
        A = np.full(len(y), np.nan)
        A[self.w - 1:] = np.convolve(y, np.ones(self.w) / self.w, mode="valid")
        return A, y - A

    def fit(self, y_train):
        A, D = self._split(y_train)
        k = self.w - 1
        self.lin = AutoARIMA(subset=False, verbose=False).fit(A[k:])      # smooth series needs many terms
        self.ann = NeuralNetwork(self.p, self.q, seed=self.seed).fit(D[k:])
        return self

    def predict_one_step(self, y):
        A, D = self._split(y)
        k = self.w - 1
        out = np.full(len(y), np.nan)
        out[k:] = self.lin.predict_one_step(A[k:]) + self.ann.predict_one_step(D[k:])
        return out


# ------------------------------------------------------------------------------------------- E4
class GatedHybrid:
    """y_hat = L_hat + G_t * N_hat_t, N_hat from the residual lags (as in Zhang) and
    G_t = logistic(g0 + g1 * y_{t-1} + g2 * L_hat_t): how much of the ANN correction is used depends on the
    state of the series. Network and gate are trained together on the training MSE."""

    def __init__(self, n_inputs, n_hidden, seed=0):
        self.p, self.q, self.seed = n_inputs, n_hidden, seed

    def _design(self, y, L):
        e = self.sc.diff(y - L)
        Xe, t = _rows([(e, k) for k in range(1, self.p + 1)], self.start, len(y))
        Xg = np.column_stack([np.ones(len(t)), self.sc.level(y[t - 1]), self.sc.level(L[t])])
        return Xe, Xg, e, t

    def _predict(self, w, Xe, Xg):
        n_net = self.p * self.q + 2 * self.q + 1
        N = self.nn._forward(w[:n_net], Xe)[1]
        return sigmoid(Xg @ w[n_net:]) * N

    def fit(self, y_train):
        self.lin = AutoARIMA(verbose=False).fit(y_train)
        self.sc = _Scaler(y_train)
        self.start = self.lin.max_lag + self.p
        Xe, Xg, e, t = self._design(y_train, self.lin.predict_one_step(y_train))
        self.nn = NeuralNetwork(self.p, self.q, seed=self.seed)
        w0 = np.concatenate([self.nn.rng.uniform(-0.5, 0.5, self.p * self.q + 2 * self.q + 1), np.zeros(3)])
        self.w = grg2_minimize(lambda w: np.mean((self._predict(w, Xe, Xg) - e[t]) ** 2), w0).x
        return self

    def predict_one_step(self, y):
        L = self.lin.predict_one_step(y)
        Xe, Xg, _, t = self._design(y, L)
        out = np.full(len(y), np.nan)
        out[t] = L[t] + self._predict(self.w, Xe, Xg) * self.sc.rng
        return out


# ------------------------------------------------------------------------------------------- E5
def _pure_ar(y_train, d, max_order=12):
    """AR(k) on the d-times differenced training series, k = 0..max_order chosen by SBC on a common sample.
    Returns (lags, constant, coefficients) with the constant in the units of the series."""
    w = np.diff(y_train, n=d) if d else np.asarray(y_train, float)
    max_order = min(max_order, max(0, len(w) // 4))
    target, best = w[max_order:], None
    for k in range(max_order + 1):
        X = np.column_stack([np.ones(len(target))] + [w[max_order - j:-j] for j in range(1, k + 1)])
        b = np.linalg.lstsq(X, target, rcond=None)[0]
        sbc = len(target) * np.log(max(np.mean((target - X @ b) ** 2), 1e-300)) + (k + 1) * np.log(len(target))
        if best is None or sbc < best[0]:
            best = (sbc, k, b)
    return list(range(1, best[1] + 1)), float(best[2][0]), best[2][1:]


class JointARNN:
    """y_hat = linear part + f(y_{t-1}, ..., y_{t-p}) with the AR coefficients and the network weights
    estimated together, instead of ARIMA first and the ANN on its residuals. When AutoARIMA selects a pure AR model
    with d = 0 or 1, the linear part uses its lags, d and estimates as the start. Otherwise (MA terms or d > 1) the
    linear part is the pure AR model of _pure_ar on the training data, with d capped at 1."""

    def __init__(self, n_inputs, n_hidden, seed=0):
        self.p, self.q, self.seed = n_inputs, n_hidden, seed

    def _design(self, y):
        s = self.sc.level(y)
        w = np.diff(s, prepend=np.nan) if self.d else s           # series the AR part works on
        lags = self.ar_lags
        start = max(self.p, self.d + max(lags, default=0))
        Xn, t = _rows([(s, k) for k in range(1, self.p + 1)], start, len(y))
        Xa = np.column_stack([np.ones(len(t))] + [w[t - k] for k in lags])
        base = s[t - 1] if self.d else np.zeros(len(t))           # random-walk part for d = 1
        return Xn, Xa, base, s, t

    def _predict(self, w, Xn, Xa, base):
        n_lin = Xa.shape[1]
        return base + Xa @ w[:n_lin] + self.nn._forward(w[n_lin:], Xn)[1]

    def fit(self, y_train):
        m = AutoARIMA(verbose=False).fit(y_train).model
        if m.ma_lags or m.d > 1:                                  # the linear part of E5 has to be pure AR with d <= 1
            self.d = min(m.d, 1)
            self.ar_lags, const, phi = _pure_ar(y_train, self.d)
        else:
            self.d, self.ar_lags, phi = m.d, m.ar_lags, m.phi
            const = m.theta0 if m.constant else 0.0
        self.sc = _Scaler(y_train)
        Xn, Xa, base, s, t = self._design(y_train)
        c0 = const / self.sc.rng + (0.0 if self.d else self.sc.lo * (np.sum(phi) - 1) / self.sc.rng)
        self.nn = NeuralNetwork(self.p, self.q, seed=self.seed)
        w0 = np.concatenate([[c0], phi, self.nn.rng.uniform(-0.5, 0.5, self.p * self.q + 2 * self.q + 1)])
        self.w = grg2_minimize(lambda w: np.mean((self._predict(w, Xn, Xa, base) - s[t]) ** 2), w0).x
        return self

    def predict_one_step(self, y):
        Xn, Xa, base, _, t = self._design(y)
        out = np.full(len(y), np.nan)
        out[t] = self.sc.lo + self.sc.rng * self._predict(self.w, Xn, Xa, base)
        return out


# ------------------------------------------------------------------------------------------- E6
class WeightedCombination:
    """y_hat = c + a*L_hat + b*N_hat + g*ANN_hat: the ARIMA forecast, Zhang's residual-network forecast and
    the standalone ANN forecast, with weights estimated by least squares on the training period
    (Zhang's hybrid is the special case c = 0, a = b = 1, g = 0)."""

    def __init__(self, n_inputs, n_hidden, seed=0):
        self.p, self.q, self.seed = n_inputs, n_hidden, seed

    def _parts(self, y):
        L = self.hyb.linear.predict_one_step(y)
        N = self.hyb.predict_one_step(y) - L
        return np.column_stack([np.ones(len(y)), L, N, self.ann.predict_one_step(y)])

    def fit(self, y_train):
        self.hyb = HybridModel(AutoARIMA(verbose=False), self.p, self.q, seed=self.seed).fit(y_train)
        self.ann = NeuralNetwork(self.p, self.q, seed=self.seed).fit(y_train)
        X = self._parts(y_train)
        ok = ~np.isnan(X).any(axis=1)
        self.coef, *_ = np.linalg.lstsq(X[ok], y_train[ok], rcond=None)
        return self

    def predict_one_step(self, y):
        return self._parts(y) @ self.coef


# ------------------------------------------------------------------------------------------- E7
class ValidatedCombination(WeightedCombination):
    """E6 with weights estimated out of sample. In-sample forecasts overstate the accuracy of the networks, which
    inflates their weights in E6. Here the base models are first fitted on the first (1 - val_frac) of the training
    data, the weights are estimated by least squares on their one-step forecasts for the remaining val_frac
    (data they were not fitted on), and the base models are then refitted on all training data."""

    def __init__(self, n_inputs, n_hidden, seed=0, val_frac=0.2):
        super().__init__(n_inputs, n_hidden, seed)
        self.val_frac = val_frac

    def _fit_base(self, y):
        self.hyb = HybridModel(AutoARIMA(verbose=False), self.p, self.q, seed=self.seed).fit(y)
        self.ann = NeuralNetwork(self.p, self.q, seed=self.seed).fit(y)

    def fit(self, y_train):
        n_fit = int(round(len(y_train) * (1 - self.val_frac)))
        self._fit_base(y_train[:n_fit])
        X = self._parts(y_train)[n_fit:]
        ok = ~np.isnan(X).any(axis=1)
        self.coef, *_ = np.linalg.lstsq(X[ok], y_train[n_fit:][ok], rcond=None)
        self._fit_base(y_train)
        return self
