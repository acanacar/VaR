import pandas as pd
from constants import *
from functions import *
from var_prod.var_class import *

data = pd.read_pickle(hist_pkl)

#
all_securities = set(data.columns.get_level_values(0))

#
n = 4
lis = np.random.rand(n)
lis_sum = functools.reduce(lambda a, b: a + b, lis)
portfolio_securities_weights = list(map(lambda y: y / lis_sum, lis))

#
portfolio_securities = random.sample(all_securities, n)

#
'''
col = 'Close'

df = data.loc[:, (portfolio_securities, col)]
df.columns = df.columns.droplevel(1)
df = df.dropna(axis=1, how='all').dropna(axis=0, how='any')

#
d = ValueAtRisk(.95, df, portfolio_securities_weights)
print('Covariance Matrix\n', d.covMatrix())
print('Annualized VaR(Percentage): ', d.var() * 100, '%')
print('Annualized VaR(Dollar): ', d.var(marketValue=1000000))
print('Daily VaR(Percentage): ', d.var(window=1) * 100, '%')
print('Daily VaR(Dollar): ', d.var(marketValue=1000000, window=1))
print('Annualized VaR(Percentage) - Approximation: ', d.var(Approximation=True) * 100, '%')
print('Annualized VaR(Dollar) - Approximation: ', d.var(Approximation=True, marketValue=1000000))

#
d_historical = HistoricalVaR(.95, df, portfolio_securities_weights)
# d_historical = HistoricalVaR(.95, df.as_matrix(), portfolio_securities_weights)
print('VaR(Percentage): ', d_historical.vaR(), '%')
print('Var(Dollar):', d_historical.vaR(marketValue=1000000))
print('100 day - VaR(Percentage): ', d_historical.vaR(window=100) * 100, '%')
print('100 day - Var(Dollar):', d_historical.vaR(marketValue=1000000, window=100))

#

d_monte = MonteCarloVaR(.95, df, portfolio_securities_weights)
'''
num_simulations = 10000
time_scaler = 1
calc_type = 'log'
period_interval = 252
confidence_interval = .95
price_col = 'Adj Close'
lambda_decay = .98

input_df = data.loc[:, (portfolio_securities, price_col)]
input_df.columns = input_df.columns.droplevel(1)
input_df = input_df.dropna(axis=1, how='all').dropna(axis=0, how='any')

'''
d = HistoricalVaR(interval=confidence_interval,
                  matrix=input_df,
                  weights=portfolio_securities_weights,
                  return_method=calc_type,
                  lookbackWindow=period_interval
                  )
d = HistoricalVaR(interval=confidence_interval,
                  matrix=input_df,
                  weights=portfolio_securities_weights,
                  return_method=calc_type,
                  lookbackWindow=period_interval,
                  hybrid=True,
                  lambda_decay_hist=lambda_decay
                  )
d = ValueAtRisk(interval=confidence_interval,
                matrix=input_df,
                weights=portfolio_securities_weights,
                return_method=calc_type,
                lookbackWindow=period_interval,
                timeScaler=time_scaler)
'''
d = ParametricVaREwma(interval=confidence_interval,
                      matrix=input_df,
                      weights=portfolio_securities_weights,
                      return_method=calc_type,
                      lookbackWindow=period_interval,
                      timeScaler=time_scaler,
                      lambdaDecay=lambda_decay)
'''

d = MonteCarloVaR(interval=confidence_interval,
                  matrix=input_df,
                  weights=portfolio_securities_weights,
                  return_method=calc_type,
                  lookbackWindow=period_interval,
                  numSimulations=num_simulations)
'''
d.vaR(series=True)

import numpy as np

v = np.full([2, 3], np.nan)

w = np.tile(np.transpose([1, 2, 3]), (3, 2))

from scipy import stats

q = np.arange(-3, 4, 1)

p = np.arange(.1, 1, .1)

print(stats.norm.ppf(p))
print(stats.t.cdf(q, 4))

print(stats.chi2.pdf(q, 2))

x = np.random.standard_t(df=5, size=100)  # Sampling 100 times from TDist with 5 df
y = np.random.normal(size=50)  # Sampling 50 times from a standard normal

res = stats.norm.fit(x)  # Fitting x to normal dist
print(res)

from statsmodels.stats.diagnostic import acorr_ljungbox

x = np.random.standard_t(df=5, size=500)  # Create dataset x
print(stats.jarque_bera(x))  # Jarque-Bera test
print(acorr_ljungbox(x, lags=20))  # Ljung-Box test

import statsmodels.api as sm
import matplotlib.pyplot as plt

y = np.random.standard_t(df=5, size=60)  # Create hypothetical dataset y
q1 = sm.tsa.stattools.acf(y, nlags=20)  # autocorrelation for lags 1:20
plt.bar(x=np.arange(1, len(q1)), height=q1[1:])
plt.show()
plt.close()

q2 = sm.tsa.stattools.pacf(y, nlags=20)  # partial autocorr for lags 1:20
'''Correlation between two variables can result from a mutual linear dependence on other variables (confounding). Partial autocorrelation is the autocorrelation between yt and yt–h after removing any linear dependence on y1, y2, ..., yt–h+1. The partial lag-h autocorrelation is denoted ϕ h , h .'''
plt.bar(x=np.arange(1, len(q2)), height=q2[1:])
plt.show()
plt.close()


def excess_kurtosis(x, excess=3):
    m4 = np.mean((x - np.mean(x)) ** 4)
    excess_kurt = m4 / (np.std(x) ** 4) - excess
    return excess_kurt


x = np.random.standard_t(df=5, size=60)
print(excess_kurtosis(x))

price = input_df['KOZAL.IS'].values

y = np.diff(np.log(price), n=1, axis=0)
plt.plot(y)
plt.show()
plt.close()

print(np.mean(y))
print(np.std(y, ddof=1))
print(np.min(y))
print(np.max(y))
print(stats.skew(y))
print(stats.kurtosis(y, fisher=False))
print(sm.tsa.stattools.acf(y, nlags=1))
print(sm.tsa.stattools.acf(np.square(y), nlags=1))
print(stats.jarque_bera(y))

print(acorr_ljungbox(y, lags=20))
print(acorr_ljungbox(np.square(y), lags=20))

# ACF plots and the Ljung-Box test in Python
import statsmodels.api as sm
import matplotlib.pyplot as plt

q = sm.tsa.stattools.acf(y, nlags=20)
plt.bar(x=np.arange(1, len(q)), height=q[1:])
plt.show()
plt.close()

q = sm.tsa.stattools.acf(np.square(y), nlags=20)
plt.bar(x=np.arange(1, len(q)), height=q[1:])
plt.show()
plt.close()

# QQ plots in Python
from statsmodels.graphics.gofplots import qqplot

fig1 = qqplot(y, line='q', dist=stats.norm, fit=True)
plt.show()
plt.close()

fig2 = qqplot(y, line='q', dist=stats.t, distargs=(5,), fit=True)
plt.show()
plt.close()

p = input_df.values

y = np.diff(np.log(p), n=1, axis=0)
np.corrcoef(y, rowvar=False)

# Chapter 2: Univariate Volatility Modelling¶
price = input_df['KOZAL.IS'].values
y = np.diff(np.log(price), n=1, axis=0) * 100
y = y - np.mean(y)
from arch import arch_model

## ARCH(1)
am = arch_model(y, mean='Zero', vol='Garch', p=1, o=0, q=0, dist='Normal')
am.fit(update_freq=5)

## ARCH(4)
am = arch_model(y, mean='Zero', vol='Garch', p=4, o=0, q=0, dist='Normal')
am.fit(update_freq=5)

## GARCH(4,1)
am = arch_model(y, mean='Zero', vol='Garch', p=4, o=0, q=1, dist='Normal')
am.fit(update_freq=5)

## GARCH(1,1)
am = arch_model(y, mean='Zero', vol='Garch', p=1, o=0, q=1, dist='Normal')
am.fit(update_freq=5)

## t-GARCH(1,1)
am = arch_model(y, mean='Zero', vol='Garch', p=1, o=0, q=1, dist='StudentsT')
am.fit(update_freq=5)

# Chapter 3: Multivariate Volatility Models¶

p = input_df.values
p = p[:, [0, 1]]

y = np.diff(np.log(p), n=1, axis=0) * 100

y[:, 0] = y[:, 0] - np.mean(y[:, 0])
y[:, 1] = y[:, 1] - np.mean(y[:, 1])
T = len(y[:, 0])

EWMA = np.full([T, 3], np.nan)

lmbda = 0.94

S = np.cov(y, rowvar=False)

EWMA[0,] = S.flatten()[[0, 3, 1]]

for i in range(1, T):
    S = lmbda * S + (1 - lmbda) * np.transpose(np.asmatrix(y[i - 1])) * np.asmatrix(y[i - 1])
    EWMA[i,] = [S[0, 0], S[1, 1], S[0, 1]]

EWMArho = np.divide(EWMA[:, 2], np.sqrt(np.multiply(EWMA[:, 0], EWMA[:, 1])))

print(EWMArho)

# Chapter 4: Risk Measures

from scipy import stats

p = [0.5, 0.1, 0.05, 0.025, 0.01, 0.001]
VaR = stats.norm.ppf(p)
ES = stats.norm.pdf(stats.norm.ppf(p)) / p

# Chapter 5: Implementing Risk Forecasts¶
p = input_df.values
p = p[:, [0, 1]]

y1 = np.diff(np.log(p[:, 0]), n=1, axis=0)
y2 = np.diff(np.log(p[:, 1]), n=1, axis=0)
y1 = y1[len(y1) - 4100:]
y2 = y2[len(y2) - 4100:]

y = np.stack([y1, y2], axis=1)

T = len(y1)
value = 1000
p = .01

# Univariate HS

ys = np.sort(y1)  # sort returns
op = int(T * p)  # p percent smallest
VaR1 = -ys[op - 1] * value

print(VaR1)

# Multivariate HS in Python
w = [[0.3], [0.7]]  # vector of portfolio weights
yp = np.squeeze(np.matmul(y, w))  # portfolio returns
yps = np.sort(yp)
VaR2 = -yps[op - 1] * value

print(VaR2)

# Univariate ES

ES1 = -np.mean(ys[:op]) * value

# Normal VaR

sigma = np.std(y1, ddof=1)  # estimate volatility
VaR3 = -sigma * stats.norm.ppf(p) * value

# Portfolio normal VaR

## portfolio volatility
sigma = np.sqrt(np.mat(np.transpose(w)) * np.mat(np.cov(y, rowvar=False)) * np.mat(w))[0, 0]
## Note: [0,0] is to pull the first element of the matrix out as a float
VaR4 = -sigma * stats.norm.ppf(p) * value

# Student-t VaR

scy1 = y1 * 100  # scale the returns
res = stats.t.fit(scy1)

sigma = res[2] / 100  # rescale volatility
nu = res[0]

VaR5 = -sigma * stats.t.ppf(p, nu) * value

print(VaR5)

#Normal ES

sigma = np.std(y1, ddof=1)
ES2 = sigma * stats.norm.pdf(stats.norm.ppf(p)) / p * value

print(ES2)


# Direct integration ES

from scipy.integrate import quad

VaR = -stats.norm.ppf(p)
integrand = lambda q: q * stats.norm.pdf(q)
ES = -sigma * quad(integrand, -np.inf, -VaR)[0] / p * value

print(ES)