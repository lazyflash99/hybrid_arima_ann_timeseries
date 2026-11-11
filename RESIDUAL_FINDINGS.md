# Residual findings

Generated from `results/3_residuals_and_decomposition/3_residual_tests.csv` by `residual_diagnostics.py`; tests at the 5% level.

## Sunspot

ARIMA training residuals show no significant autocorrelation (Ljung-Box p = 0.713). Nonlinearity tests on them give BDS p = 0.000 and neural-network test p = 0.001, so nonlinearity is detected. ARCH-LM p = 0.000 (evidence of changing variance) and Jarque-Bera p = 0.000 (not normal). The size of the residuals rises significantly with the level (Spearman 0.23, p = 0.001). The hybrid lowered the 10-seed mean test MSE (306.1 for ARIMA, 289.7 for the hybrid).

## Lynx

ARIMA training residuals show no significant autocorrelation (Ljung-Box p = 0.342). Nonlinearity tests on them give BDS p = 0.813 and neural-network test p = 0.820, so nonlinearity is not detected. ARCH-LM p = 0.506 (no evidence of changing variance) and Jarque-Bera p = 0.468 (normality not rejected). The size of the residuals does not rise significantly with the level (Spearman -0.15, p = 0.161). The hybrid did not lower the 10-seed mean test MSE (0.02421 for ARIMA, 0.0246 for the hybrid).

## Exchange rate

ARIMA training residuals show no significant autocorrelation (Ljung-Box p = 0.432). Nonlinearity tests on them give BDS p = 0.047 and neural-network test p = 0.065, so nonlinearity is detected. ARCH-LM p = 0.000 (evidence of changing variance) and Jarque-Bera p = 0.000 (not normal). The size of the residuals does not rise significantly with the level (Spearman -0.03, p = 0.505). The hybrid did not lower the 10-seed mean test MSE (4.2e-05 for ARIMA, 4.232e-05 for the hybrid).

## AirPassengers

ARIMA training residuals show significant autocorrelation (Ljung-Box p = 0.000). Nonlinearity tests on them give BDS p = 0.002 and neural-network test p = 0.466, so nonlinearity is detected. ARCH-LM p = 0.064 (no evidence of changing variance) and Jarque-Bera p = 0.125 (normality not rejected). The size of the residuals does not rise significantly with the level (Spearman -0.23, p = 0.017). The hybrid did not lower the 10-seed mean test MSE (0.0003442 for ARIMA, 0.0005023 for the hybrid).

## Nile

ARIMA training residuals show no significant autocorrelation (Ljung-Box p = 0.344). Nonlinearity tests on them give BDS p = 0.272 and neural-network test p = 0.344, so nonlinearity is not detected. ARCH-LM p = 0.478 (no evidence of changing variance) and Jarque-Bera p = 0.684 (normality not rejected). The size of the residuals does not rise significantly with the level (Spearman 0.06, p = 0.586). The hybrid lowered the 10-seed mean test MSE (1.692e+04 for ARIMA, 1.567e+04 for the hybrid).

## Daily min temp

ARIMA training residuals show no significant autocorrelation (Ljung-Box p = 0.987). Nonlinearity tests on them give BDS p = 0.000 and neural-network test p = 0.048, so nonlinearity is detected. ARCH-LM p = 0.000 (evidence of changing variance) and Jarque-Bera p = 0.337 (normality not rejected). The size of the residuals does not rise significantly with the level (Spearman 0.02, p = 0.588). The hybrid did not lower the 10-seed mean test MSE (5.721 for ARIMA, 5.772 for the hybrid).

## FX JPY

ARIMA training residuals show no significant autocorrelation (Ljung-Box p = 0.724). Nonlinearity tests on them give BDS p = 0.004 and neural-network test p = 0.040, so nonlinearity is detected. ARCH-LM p = 0.000 (evidence of changing variance) and Jarque-Bera p = 0.000 (not normal). The size of the residuals does not rise significantly with the level (Spearman 0.02, p = 0.689). The hybrid lowered the 10-seed mean test MSE (3.121e-05 for ARIMA, 3.11e-05 for the hybrid).

## Sunspot sqrt

ARIMA training residuals show no significant autocorrelation (Ljung-Box p = 0.562). Nonlinearity tests on them give BDS p = 0.091 and neural-network test p = 0.017, so nonlinearity is detected. ARCH-LM p = 0.022 (evidence of changing variance) and Jarque-Bera p = 0.152 (normality not rejected). The size of the residuals does not rise significantly with the level (Spearman -0.08, p = 0.229). The hybrid lowered the 10-seed mean test MSE (1.288 for ARIMA, 1.223 for the hybrid).

## Synthetic additive

ARIMA training residuals show no significant autocorrelation (Ljung-Box p = 0.520). Nonlinearity tests on them give BDS p = 0.094 and neural-network test p = 0.000, so nonlinearity is detected. ARCH-LM p = 0.000 (evidence of changing variance) and Jarque-Bera p = 0.004 (not normal). The size of the residuals does not rise significantly with the level (Spearman -0.09, p = 0.114). The hybrid did not lower the 10-seed mean test MSE (3.975 for ARIMA, 3.983 for the hybrid).

## Synthetic multiplicative

ARIMA training residuals show no significant autocorrelation (Ljung-Box p = 0.491). Nonlinearity tests on them give BDS p = 0.875 and neural-network test p = 0.742, so nonlinearity is not detected. ARCH-LM p = 0.182 (no evidence of changing variance) and Jarque-Bera p = 0.001 (not normal). The size of the residuals does not rise significantly with the level (Spearman nan, p = nan). The hybrid did not lower the 10-seed mean test MSE (9.046 for ARIMA, 9.251 for the hybrid).

## Mackey-Glass

ARIMA training residuals show no significant autocorrelation (Ljung-Box p = 0.086). Nonlinearity tests on them give BDS p = 0.004 and neural-network test p = 0.042, so nonlinearity is detected. ARCH-LM p = 0.040 (evidence of changing variance) and Jarque-Bera p = 0.042 (not normal). The size of the residuals does not rise significantly with the level (Spearman 0.04, p = 0.431). The hybrid lowered the 10-seed mean test MSE (0.006399 for ARIMA, 0.006304 for the hybrid).
