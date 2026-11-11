"""
The Extension: one neural network forecasts y_t from past values, the ARIMA forecast and past ARIMA errors,

    y_hat_t = f(y_{t-1}, ..., y_{t-p}, L_hat_t, e_{t-1}, ..., e_{t-q}),

instead of adding a separate network forecast of the ARIMA error to the ARIMA forecast as Zhang (2003) does.
Related earlier work is listed in REFERENCES.md.
"""
import numpy as np
import scipy.optimize


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class ExtensionHybrid:
    """
    Parameters
    ----------
    linear_model : object
        Linear model instance with `fit(y_train)` and `predict_one_step(y_all)`.
        (e.g., arima_model.ARIMA).
    n_series_lags : int, default=4
        Number of past lags of the actual series (y_{t-1}, ..., y_{t-p}).
    n_residual_lags : int, default=2
        Number of past lags of the linear residuals (e_{t-1}, ..., e_{t-q}).
    n_hidden : int, default=4
        Number of hidden neurons in the neural network layer.
    include_linear_pred : bool, default=True
        Whether to include the contemporaneous linear prediction L_hat_t as an input.
    weight_decay : float, default=1e-5
        L2 regularization penalty.
    n_restarts : int, default=20
        Number of random starts; the fit with the lowest training loss is kept.
    max_iter : int, default=1500
        Maximum optimization iterations per restart.
    seed : int, default=0
        Random seed for reproducibility.

    Options for values outside the training range (all off by default):
    output : "direct" (default) or "skip"
        "direct": the network outputs y_t on the scaled range. The output layer is linear, so forecasts can
        exceed the training range. The limitation is inputs outside that range, where the hidden units saturate.
        "skip": the network outputs a correction and y_hat_t = L_hat_t + correction, so the ARIMA forecast
        reaches the output unchanged.
    shortcut : bool, default=False
        Adds direct linear connections from every input to the output next to the hidden layer.
    scale_margin : float, default=0.0
        Maps the training range to [margin, 1 - margin] instead of [0, 1], leaving room beyond it.
    """
    def __init__(self, linear_model, n_series_lags=4, n_residual_lags=2, n_hidden=4,
                 include_linear_pred=True,
                 weight_decay=1e-5, n_restarts=20, max_iter=1500, seed=0,
                 output="direct", shortcut=False, scale_margin=0.0):
        self.linear = linear_model
        self.p = n_series_lags
        self.q = n_residual_lags
        self.n_hidden = n_hidden
        self.include_linear_pred = include_linear_pred
        self.weight_decay = weight_decay
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.seed = seed
        if output not in ("direct", "skip"):
            raise ValueError("output must be 'direct' or 'skip'")
        self.output, self.shortcut, self.margin = output, shortcut, scale_margin
        self.span = 1.0 - 2.0 * scale_margin           # width of the scaled training range

    def _get_start_idx(self):
        m = getattr(self.linear, "max_lag", 1)
        return max(self.p, m + self.q)

    def _scale_y(self, v):
        return (v - self.y_min) / self.y_range * self.span + self.margin

    def _unscale_y(self, v):
        return (v - self.margin) / self.span * self.y_range + self.y_min

    def _scale_e(self, v):
        return v / self.y_range * self.span

    def _to_forecast(self, raw, L_hat, indices):
        """Converts the network output to a forecast on the original scale."""
        if self.output == "skip":
            return L_hat[indices] + raw / self.span * self.y_range
        return self._unscale_y(raw)

    def _build_feature_matrix(self, y_series, L_hat, e_series, start_idx, end_idx=None):
        if end_idx is None:
            end_idx = len(y_series)

        X_rows = []
        indices = np.arange(start_idx, end_idx)
        for t in indices:
            row = []
            # past values y_{t-1}, ..., y_{t-p}
            for k in range(1, self.p + 1):
                row.append(self._scale_y(y_series[t - k]))

            # ARIMA forecast L_hat_t
            if self.include_linear_pred:
                row.append(self._scale_y(L_hat[t]))

            # past ARIMA errors e_{t-1}, ..., e_{t-q}, scaled by the training range
            for k in range(1, self.q + 1):
                row.append(self._scale_e(e_series[t - k]))

            X_rows.append(row)

        X = np.array(X_rows, dtype=float)
        if self.output == "skip":                       # target: the correction to the ARIMA forecast
            y_scaled = self._scale_e(y_series[indices].astype(float) - L_hat[indices])
        else:
            y_scaled = self._scale_y(y_series[indices].astype(float))
        return X, y_scaled, indices

    def _train_scipy(self, X, y):
        n_samples, n_features = X.shape
        n_hidden = self.n_hidden
        n_net = n_features * n_hidden + n_hidden + n_hidden + 1
        n_weights = n_net + (n_features if self.shortcut else 0)
        rng = np.random.RandomState(self.seed)

        def unpack(w):
            W1 = w[:n_features * n_hidden].reshape(n_features, n_hidden)
            b1 = w[n_features * n_hidden: n_features * n_hidden + n_hidden]
            w2 = w[n_features * n_hidden + n_hidden: n_features * n_hidden + 2 * n_hidden]
            b2 = w[n_net - 1]
            v = w[n_net:]                               # direct input -> output weights (shortcut)
            return W1, b1, w2, b2, v

        def loss_and_grad(w):
            W1, b1, w2, b2, v = unpack(w)
            H = sigmoid(X @ W1 + b1)
            pred = H @ w2 + b2 + (X @ v if len(v) else 0.0)
            err = pred - y
            loss = np.mean(err ** 2) + self.weight_decay * np.sum(w ** 2)

            d = 2.0 * err / n_samples
            g_w2 = H.T @ d
            g_b2 = d.sum()
            dZ = np.outer(d, w2) * H * (1.0 - H)
            g_W1 = X.T @ dZ
            g_b1 = dZ.sum(axis=0)
            g_v = X.T @ d if len(v) else np.zeros(0)
            grad = np.concatenate([g_W1.ravel(), g_b1, g_w2, [g_b2], g_v]) + 2.0 * self.weight_decay * w
            return loss, grad

        best_loss, best_w = np.inf, None
        for _ in range(self.n_restarts):
            w0 = rng.uniform(-0.5, 0.5, n_weights)
            res = scipy.optimize.minimize(
                loss_and_grad, w0, jac=True, method="L-BFGS-B",
                options={"maxiter": self.max_iter}
            )
            if res.fun < best_loss:
                best_loss, best_w = res.fun, res.x

        self.w = best_w
        self.unpack = unpack

    def fit(self, y_train):
        """Fits ARIMA, then the network, on the training data only."""
        self.y_min = y_train.min()
        self.y_max = y_train.max()
        self.y_range = max(1e-8, self.y_max - self.y_min)

        self.linear.fit(y_train)
        L_hat = self.linear.predict_one_step(y_train)
        e_train = y_train - L_hat

        self.start_idx = self._get_start_idx()

        X_tr, y_tr_scaled, _ = self._build_feature_matrix(
            y_train, L_hat, e_train, self.start_idx, len(y_train)
        )

        self._train_scipy(X_tr, y_tr_scaled)

        # training error, kept for reporting
        preds_scaled = self._predict_raw(X_tr)
        preds = self._to_forecast(preds_scaled, L_hat, np.arange(self.start_idx, len(y_train)))
        y_tr_actual = y_train[self.start_idx:]
        self.train_mse = np.mean((y_tr_actual - preds) ** 2)
        return self

    def _predict_raw(self, X):
        W1, b1, w2, b2, v = self.unpack(self.w)
        H = sigmoid(X @ W1 + b1)
        return H @ w2 + b2 + (X @ v if len(v) else 0.0)

    def predict_one_step(self, y_all):
        """One-step forecasts for the whole series (training + test); the first start_idx entries are NaN."""
        L_hat = self.linear.predict_one_step(y_all)
        e_all = y_all - L_hat

        X_all, _, indices = self._build_feature_matrix(
            y_all, L_hat, e_all, self.start_idx, len(y_all)
        )

        y_hat_scaled = self._predict_raw(X_all)
        y_hat = self._to_forecast(y_hat_scaled, L_hat, indices)

        full_preds = np.full(len(y_all), np.nan)
        full_preds[indices] = y_hat
        return full_preds
