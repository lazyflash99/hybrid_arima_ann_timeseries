"""
Nonlinear part of the paper: single-hidden-layer feedforward network (Section 2.2, Eqs. 2-4).

  y_t = a0 + sum_j a_j * g( b0j + sum_i b_ij * y_{t-i} ) + e_t ,   g(x) = 1 / (1 + exp(-x))   (Eqs. 2, 3)

p = number of input nodes (lags), q = number of hidden nodes, 1 linear output node
(one output node -> one-step-ahead forecasting).

Training (Section 2.2): "the parameters are estimated such that an overall accuracy criterion such as
the mean squared error is minimized ... with some efficient nonlinear optimization algorithms other than
the basic backpropagation training algorithm", namely GRG2. The objective here is the plain training MSE.
Only the training sample is used; the test sample is held out (Section 2.2).

Necessary substitutes for what the paper does not specify or what is not available:
  * GRG2 is not freely available. On this unconstrained problem GRG2 takes quasi-Newton (BFGS) steps,
    so scipy's BFGS with the analytic gradient is used, stopped with GRG2's default termination test:
    the fractional change of the objective stays below EPSTOP = 1e-4 for NSTOP = 3 consecutive iterations.
  * one training run from random starting weights drawn from U(-0.5, 0.5) (the paper gives no initialisation).
  * inputs/target are linearly rescaled to [0, 1] with the training min/max. This is an affine change of
    variables that the weights b_0j, b_ij, a_0, a_j absorb exactly, so the model is still Eq. 2.
"""
import numpy as np
from scipy.optimize import minimize

EPSTOP, NSTOP = 1e-4, 3      # GRG2 default termination parameters


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))   # clip to avoid overflow in exp


def grg2_minimize(fun, w0, jac=False, max_iter=2000):
    """BFGS stopped by the GRG2 termination test (fractional objective change < EPSTOP for NSTOP iterations).
    jac=True: fun returns (loss, grad); jac=False: gradient by finite differences."""
    history = [fun(w0)[0] if jac else fun(w0)]

    def grg2_stop(intermediate_result):
        history.append(intermediate_result.fun)
        recent = history[-(NSTOP + 1):]
        if len(recent) == NSTOP + 1 and all(abs(b - a) / max(abs(a), 1e-12) < EPSTOP
                                            for a, b in zip(recent[:-1], recent[1:])):
            raise StopIteration

    return minimize(fun, w0, jac=jac, method="BFGS", callback=grg2_stop, options={"maxiter": max_iter})


def lag_matrix(series, p, start):
    """Row for time t = [s[t-1], s[t-2], ..., s[t-p]] ; target = s[t], for t = start..end."""
    idx = np.arange(start, len(series))
    X = np.column_stack([series[idx - k] for k in range(1, p + 1)])
    return X, series[idx]


class NeuralNetwork:
    def __init__(self, n_inputs, n_hidden, max_iter=2000, seed=0):
        self.p, self.q = n_inputs, n_hidden
        self.max_iter = max_iter
        self.rng = np.random.RandomState(seed)

    # ---- weights are kept in one flat vector w ------------------------------------------
    def _unpack(self, w):
        p, q = self.p, self.q
        W1 = w[:p * q].reshape(p, q)          # b_ij  (input -> hidden)
        b1 = w[p * q:p * q + q]               # b_0j  (hidden biases)
        w2 = w[p * q + q:p * q + 2 * q]       # a_j   (hidden -> output)
        b2 = w[-1]                            # a_0   (output bias)
        return W1, b1, w2, b2

    def _forward(self, w, X):
        W1, b1, w2, b2 = self._unpack(w)
        H = sigmoid(X @ W1 + b1)
        return H, H @ w2 + b2

    def _mse_and_grad(self, w, X, y):
        W1, b1, w2, b2 = self._unpack(w)
        H, out = self._forward(w, X)
        err = out - y
        loss = np.mean(err ** 2)
        d = 2 * err / len(y)                  # dLoss/dOut
        g_w2 = H.T @ d
        g_b2 = d.sum()
        dZ = np.outer(d, w2) * H * (1 - H)    # through the logistic function
        g_W1 = X.T @ dZ
        g_b1 = dZ.sum(axis=0)
        grad = np.concatenate([g_W1.ravel(), g_b1, g_w2, [g_b2]])
        return loss, grad

    def _train(self, X, y):
        """One BFGS run from random weights, stopped by the GRG2 termination test."""
        n_weights = self.p * self.q + 2 * self.q + 1
        w0 = self.rng.uniform(-0.5, 0.5, n_weights)
        res = grg2_minimize(lambda w: self._mse_and_grad(w, X, y), w0, jac=True, max_iter=self.max_iter)
        self.n_iter = res.nit
        return res.x

    # ---- scaling: logistic units work best on data in [0, 1] -----------------------------
    def _scale(self, s):
        return (s - self.lo) / (self.hi - self.lo)

    def _unscale(self, s):
        return s * (self.hi - self.lo) + self.lo

    def fit(self, train_series):
        self.lo, self.hi = train_series.min(), train_series.max()
        s = self._scale(train_series)
        X, y = lag_matrix(s, self.p, start=self.p)
        self.w = self._train(X, y)

        # diagnostics
        _, out = self._forward(self.w, X)
        self.train_mse = np.mean((self._unscale(out) - self._unscale(y)) ** 2)
        self.max_abs_weight = np.abs(self.w).max()
        return self

    def predict_one_step(self, series):
        """One-step-ahead predictions using actual past values. NaN for the first p entries."""
        s = self._scale(series)
        X, _ = lag_matrix(s, self.p, start=self.p)
        _, out = self._forward(self.w, X)
        y_hat = np.full(len(series), np.nan)
        y_hat[self.p:] = self._unscale(out)
        return y_hat
