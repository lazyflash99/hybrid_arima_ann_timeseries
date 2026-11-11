"""
Hybrid methodology (Section 3):

  y_t = L_t + N_t                               (Eq. 5)
  Step 1: ARIMA models the linear part  -> L_hat_t
  Step 2: residuals e_t = y_t - L_hat_t         (Eq. 6)
          ANN models residuals: e_t = f(e_{t-1},...,e_{t-n}) + eps_t   (Eq. 7) -> N_hat_t
  Final : y_hat_t = L_hat_t + N_hat_t           (Eq. 8)
"""
import numpy as np
from ann_model import NeuralNetwork


class HybridModel:
    def __init__(self, linear_model, n_inputs, n_hidden, **ann_kwargs):
        self.linear = linear_model
        self.ann = NeuralNetwork(n_inputs, n_hidden, **ann_kwargs)

    def fit(self, y_train):
        self.linear.fit(y_train)                                   # Step 1
        L_hat = self.linear.predict_one_step(y_train)
        e = y_train - L_hat                                        # Eq. 6
        self.first = self.linear.max_lag                           # residuals exist from here on
        self.ann.fit(e[self.first:])                               # Step 2
        return self

    def predict_one_step(self, y_all):
        """y_all = training + test series. Residuals of the test period are computed from
        the actual test values (one-step-ahead), consistent with the lagged inputs of the other models."""
        L_hat = self.linear.predict_one_step(y_all)
        e = (y_all - L_hat)[self.first:]
        N_hat = np.full(len(y_all), np.nan)
        N_hat[self.first:] = self.ann.predict_one_step(e)
        return L_hat + N_hat                                       # Eq. 8
