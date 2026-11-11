"""
Linear part of the paper: ARIMA model (Section 2.1, Eq. 1), built with the Box-Jenkins steps

  y_t = theta_0 + phi_1 y_{t-1} + ... + phi_p y_{t-p} + eps_t - theta_1 eps_{t-1} - ... - theta_q eps_{t-q}   (Eq. 1)

applied to the d-times differenced series w_t = (1 - B)^d y_t.

  * Identification     : sample ACF / PACF of the (differenced) series           -> acf(), pacf()
  * Estimation         : "parameters are estimated such that an overall measure of errors is minimized ...
                          with a nonlinear optimization procedure". The paper fits every ARIMA model in
                          SAS/ETS (Section 4.2), whose PROC ARIMA default is conditional least squares (CLS):
                          minimise sum(eps_t^2) over all observations, with unobserved pre-sample values set
                          to the mean (deviation 0) and pre-sample errors set to 0. Solved with Levenberg-Marquardt.
  * Diagnostic checking: Ljung-Box chi-square test of the residual autocorrelations at lags 6, 12, 18, 24
                          (the "Autocorrelation Check of Residuals" printed by SAS)  -> ljung_box()

Model selection: AutoARIMA / select_arima() automate the Box-Jenkins cycle on the training data with the
paper's rule (Section 4.2): the most parsimonious model among all ARIMA models judged adequate by residual
analysis. ARIMA(...) with a fixed order reproduces the models the paper reports.

Models finally selected in Section 4.2:
  * Sunspot       : subset AR model of order 9        -> ARIMA(p=[1, 2, 9], d=0, q=0)
  * Lynx          : subset AR model of order 12       -> ARIMA(p=[1, 2, 4, 6, 9, 10, 11, 12], d=0, q=0)
  * Exchange rate : random walk y_t = y_{t-1} + eps_t -> ARIMA(p=0, d=1, q=0, constant=False)
All forecasts are one step ahead, as in the paper:
every forecast uses the actual past observations.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter
from scipy.stats import chi2


def _as_lags(order):
    """An order is either an int k (lags 1..k) or an explicit list of lags (subset model)."""
    if np.isscalar(order):
        return list(range(1, int(order) + 1))
    return sorted(int(k) for k in order)


def _lag_poly(lags, coefs):
    """1 - sum_k c_k B^k as a coefficient array (index = power of B)."""
    poly = np.zeros(max(lags, default=0) + 1)
    poly[0] = 1.0
    for k, c in zip(lags, coefs):
        poly[k] = -c
    return poly


def acf(x, nlags):
    x = np.asarray(x, float) - np.mean(x)
    denom = x @ x
    return np.array([1.0] + [x[k:] @ x[:-k] / denom for k in range(1, nlags + 1)])


def pacf(x, nlags):
    """Durbin-Levinson recursion on the sample ACF."""
    r = acf(x, nlags)
    out, phi = [1.0], np.zeros(0)
    for k in range(1, nlags + 1):
        a = (r[k] - phi @ r[1:k][::-1]) / (1.0 - phi @ r[1:k])
        phi = np.concatenate([phi - a * phi[::-1], [a]])
        out.append(a)
    return np.array(out)


def ljung_box(resid, n_params, lags=(6, 12, 18, 24)):
    """Returns [(lag, Q, df, p-value)]; lags with df <= 0 are skipped."""
    n = len(resid)
    r = acf(resid, max(lags))
    rows = []
    for m in lags:
        df = m - n_params
        if df <= 0:
            continue
        Q = n * (n + 2) * np.sum(r[1:m + 1] ** 2 / (n - np.arange(1, m + 1)))
        rows.append((m, Q, df, chi2.sf(Q, df)))
    return rows


class ARIMA:
    """
    p, q     : int (all lags 1..p) or list of lags (subset model, e.g. [1, 2, 9])
    d        : order of differencing
    constant : include the constant term theta_0 of Eq. 1 (estimated as the mean mu of w_t,
               theta_0 = mu * (1 - sum(phi)), as SAS does)
    """

    def __init__(self, p=0, d=0, q=0, constant=True, verbose=True):
        self.ar_lags, self.ma_lags = _as_lags(p), _as_lags(q)
        self.d, self.constant, self.verbose = d, constant, verbose
        # first index of y with a forecast built only from observed data (no pre-sample values)
        self.max_lag = d + max(self.ar_lags, default=0)

    def _difference(self, y):
        return np.diff(y, n=self.d) if self.d else np.asarray(y, float)

    def _unpack(self, params):
        i = int(self.constant)
        mu = params[0] if self.constant else 0.0
        phi = params[i:i + len(self.ar_lags)]
        theta = params[i + len(self.ar_lags):]
        return mu, phi, theta

    def _residuals(self, params, w):
        """eps_t of Eq. 1 for every t, pre-sample deviations and errors set to 0 (CLS)."""
        mu, phi, theta = self._unpack(params)
        return lfilter(_lag_poly(self.ar_lags, phi), _lag_poly(self.ma_lags, theta), w - mu)

    def _start_values(self, w):
        mu = np.mean(w) if self.constant else 0.0
        phi = np.zeros(len(self.ar_lags))
        if self.ar_lags:
            m = max(self.ar_lags)
            X = np.array([[w[t - k] - mu for k in self.ar_lags] for t in range(m, len(w))])
            phi, *_ = np.linalg.lstsq(X, w[m:] - mu, rcond=None)
        return np.concatenate([[mu] if self.constant else [], phi, np.zeros(len(self.ma_lags))])

    # ---------------------------------------------------------------- Box-Jenkins steps

    def fit(self, y_train):
        w = self._difference(y_train)
        x0 = self._start_values(w)
        if len(x0):
            sol = least_squares(self._residuals, x0, args=(w,), method="lm")
            self.params = sol.x
            jac = sol.jac
        else:
            self.params, jac = x0, None
        self.resid = self._residuals(self.params, w)
        self.mu, self.phi, self.theta = self._unpack(self.params)
        self.theta0 = self.mu * (1.0 - np.sum(self.phi))
        n_par = len(self.params)
        self.sigma2 = self.resid @ self.resid / max(1, len(w) - n_par)
        with np.errstate(invalid="ignore"):
            self.stderr = (np.sqrt(np.diag(self.sigma2 * np.linalg.pinv(jac.T @ jac)))
                           if jac is not None else np.zeros(0))
        self.diagnostics = ljung_box(self.resid, len(self.ar_lags) + len(self.ma_lags))
        if self.verbose:
            self.summary()
        return self

    def summary(self):
        print(f"   ARIMA(p={self.ar_lags}, d={self.d}, q={self.ma_lags}), constant={self.constant}")
        names = (["MU"] if self.constant else []) + [f"AR{k}" for k in self.ar_lags] \
            + [f"MA{k}" for k in self.ma_lags]
        for name, b, se in zip(names, self.params, self.stderr):
            print(f"     {name:5} {b: .6f}  (s.e. {se:.6f}, t = {b / se: .2f})")
        if self.constant:
            print(f"     theta_0 (Eq. 1 constant) = {self.theta0:.6f}")
        print(f"     sigma^2 = {self.sigma2:.6g}")
        for m, Q, df, pv in self.diagnostics:
            print(f"     Ljung-Box to lag {m:2}: Q = {Q:7.2f}, df = {df:2}, p = {pv:.3f}")

    # ---------------------------------------------------------------- forecasting
    def predict_one_step(self, y):
        """Returns array of same length as y; first max_lag entries are NaN.
        The one-step forecast is y_t - eps_t, with eps_t from Eq. 1 run over the actual series
        (the filter is causal, so each forecast uses only observations before t)."""
        y = np.asarray(y, float)
        eps = self._residuals(self.params, self._difference(y))
        y_hat = np.full(len(y), np.nan)
        y_hat[self.d:] = y[self.d:] - eps
        y_hat[:self.max_lag] = np.nan
        return y_hat


# -------------------------------------------------------------------- automatic Box-Jenkins selection
def _roots_outside_unit_circle(lags, coefs):
    if not len(lags):
        return True
    return np.all(np.abs(np.roots(_lag_poly(lags, coefs)[::-1])) > 1.0 + 1e-6)


def choose_d(y, alpha=0.05, max_d=2):
    """Identification of d: difference until the augmented Dickey-Fuller test rejects a unit root."""
    import warnings
    from statsmodels.tsa.stattools import adfuller
    for d in range(max_d + 1):
        w = np.diff(y, n=d) if d else y
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)     # adfuller return-type notice
            p_value = adfuller(w, regression="c", autolag="AIC")[1]
        if p_value < alpha:
            return d
    return max_d


_SELECTION_CACHE = {}
DEFAULT_SUBSET = True          # AutoARIMA search: subset lags (True) or full orders only; set per data set
MAX_SELECTION_FITS = 8000       # stop the search here and use the lowest-SBC model found so far


def select_arima(y_train, max_ar=12, max_ma=3, alpha=0.05, subset=True, verbose=True):
    """
    Section 4.2's rule: "the most parsimonious among all ARIMA models that are also found adequate
    judged by the residual analysis". Uses the training data only.
      * d from the ADF test (stationarity, Section 2.1)
      * candidates: any subset of AR lags 1..max_ar and MA lags 1..max_ma (subset models as in the paper),
        with or without a constant when d > 0; tried in order of increasing number of parameters
        (subset=False: only full-order models with lags 1..p and 1..q, for series that need many terms)
      * adequate = stationary and invertible, and Ljung-Box p > alpha at every lag 6, 12, 18, 24 that has df > 0
      * the smallest adequate size wins; ties are broken by the lowest SBC (printed by SAS PROC ARIMA)
      * if no candidate is adequate, the stationary and invertible model with the lowest SBC is used (with a warning)
    """
    from itertools import combinations
    key = (np.asarray(y_train, float).tobytes(), max_ar, max_ma, alpha, subset)
    if key in _SELECTION_CACHE:
        return _SELECTION_CACHE[key]

    d = choose_d(y_train, alpha)
    n = len(y_train) - d
    const_options = [True] if d == 0 else [False, True]
    n_tried, fallback = 0, []
    for k in range(0, max_ar + max_ma + 2):
        if n_tried > MAX_SELECTION_FITS:
            break
        adequate = []
        for constant in const_options:
            n_arma = k - int(constant)
            for a in range(0, min(n_arma, max_ar) + 1):
                b = n_arma - a
                if b < 0 or b > max_ma:
                    continue
                ar_sets = combinations(range(1, max_ar + 1), a) if subset else [range(1, a + 1)]
                for ar in ar_sets:
                    ma_sets = combinations(range(1, max_ma + 1), b) if subset else [range(1, b + 1)]
                    for ma in ma_sets:
                        m = ARIMA(p=list(ar), d=d, q=list(ma), constant=constant, verbose=False).fit(y_train)
                        n_tried += 1
                        if not (_roots_outside_unit_circle(m.ar_lags, m.phi)
                                and _roots_outside_unit_circle(m.ma_lags, m.theta)):
                            continue
                        sbc = n * np.log(m.resid @ m.resid / n) + k * np.log(n)
                        fallback.append((sbc, m))
                        if not m.diagnostics or any(pv <= alpha for *_, pv in m.diagnostics):
                            continue
                        adequate.append((sbc, m))
        if adequate:
            best = min(adequate, key=lambda t: t[0])[1]
            if verbose:
                print(f"   Box-Jenkins selection: d = {d} (ADF), {n_tried} models tried, "
                      f"{len(adequate)} adequate with {k} parameters, lowest SBC chosen")
            _SELECTION_CACHE[key] = best
            return best
    if not fallback:
        raise RuntimeError("no stationary and invertible ARIMA model found")
    best = min(fallback, key=lambda t: t[0])[1]
    if verbose:
        print(f"   WARNING: no candidate passed the Ljung-Box check ({n_tried} tried); lowest-SBC model used: "
              f"ARIMA(p={best.ar_lags}, d={d}, q={best.ma_lags})")
    _SELECTION_CACHE[key] = best
    return best


class AutoARIMA:
    """Linear model with the same interface as ARIMA, whose order is chosen by select_arima()."""

    def __init__(self, max_ar=12, max_ma=3, alpha=0.05, subset=None, verbose=True):
        self.kwargs = dict(max_ar=max_ar, max_ma=max_ma, alpha=alpha,
                           subset=DEFAULT_SUBSET if subset is None else subset)
        self.verbose = verbose

    def fit(self, y_train):
        self.model = select_arima(y_train, verbose=self.verbose, **self.kwargs)
        self.max_lag = self.model.max_lag
        if self.verbose:
            self.model.summary()
        return self

    def predict_one_step(self, y):
        return self.model.predict_one_step(y)
