import pandas as pd
from constants import *
from functions import *
import numpy as np

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
price_col = 'Adj Close'

input_df = data.loc[:, (portfolio_securities, price_col)]
input_df.columns = input_df.columns.droplevel(1)
input_df = input_df.dropna(axis=1, how='all').dropna(axis=0, how='any')

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

print(portfolio_securities[0])
price = input_df['TOASO.IS'].values

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
price = input_df['KCHOL.IS'].values
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

# Normal ES

sigma = np.std(y1, ddof=1)
ES2 = sigma * stats.norm.pdf(stats.norm.ppf(p)) / p * value

print(ES2)

# Direct integration ES

from scipy.integrate import quad

VaR = -stats.norm.ppf(p)
integrand = lambda q: q * stats.norm.pdf(q)
ES = -sigma * quad(integrand, -np.inf, -VaR)[0] / p * value

print(ES)

# MA normal VaR
WE = 20
for t in range(T - 5, T + 1):
    t1 = t - WE
    window = y1[t1:t]  # estimation window
    sigma = np.std(window, ddof=1)
    VaR6 = -sigma * stats.norm.ppf(p) * value
    print(VaR6)

# EWMA VaR
lmbda = 0.94
s11 = np.var(y1[0:30], ddof=1)  # initial variance

for t in range(1, T):
    s11 = lmbda * s11 + (1 - lmbda) * y1[t - 1] ** 2

VaR7 = -np.sqrt(s11) * stats.norm.ppf(p) * value

print(VaR7)

# Two-asset EWMA VaR

s = np.cov(y, rowvar=False)
for t in range(1, T):
    s = lmbda * s + (1 - lmbda) * np.transpose(np.asmatrix(y[t - 1, :])) * np.asmatrix(y[t - 1, :])

sigma = np.sqrt((np.transpose(w) * s * w)[0, 0])

VaR8 = -sigma * stats.norm.ppf(p) * value

# GARCH VaR

from arch import arch_model

am = arch_model(y1, mean='Zero', vol='Garch', p=1, o=0, q=1, dist='Normal')
res = am.fit(update_freq=5)
omega, alpha, beta = res.params[0], res.params[1], res.params[2]
## computing sigma2 for t+1
sigma2 = omega + alpha * y1[T - 1] ** 2 + beta * res.conditional_volatility[-1] ** 2

VaR9 = -np.sqrt(sigma2) * stats.norm.ppf(p) * value
print(VaR9)

# Chapter 8: Backtesting and Stress Testing¶

price = input_df['TOASO.IS'].values
y = np.diff(np.log(price), n=1, axis=0)  # get returns
T = len(y)  # number of obs for y
WE = 1000  # estimation window length
p = 0.01  # probability
l1 = int(WE * p)  # HS observation
value = 1  # portfolio value
VaR = np.full([T, 4], np.nan)  # matrix for forecasts

## EWMA setup

lmbda = 0.94
s11 = np.var(y[1:30])

for t in range(1, WE):
    s11 = lmbda * s11 + (1 - lmbda) * y[t - 1] ** 2

for t in range(WE, T):
    t1 = t - WE  # start of data window
    t2 = t - 1  # end of data window
    window = y[t1:t2 + 1]  # data for estimation

    s11 = lmbda * s11 + (1 - lmbda) * y[t - 1] ** 2
    VaR[t, 0] = -stats.norm.ppf(p) * np.sqrt(s11) * value  # EWMA

    VaR[t, 1] = -np.std(window, ddof=1) * stats.norm.ppf(p) * value  # MA

    ys = np.sort(window)
    VaR[t, 2] = -ys[l1 - 1] * value  # HS

    am = arch_model(window, mean='Zero', vol='Garch',
                    p=1, o=0, q=1, dist='Normal')
    res = am.fit(update_freq=0, disp='off', show_warning=False)
    par = [res.params[0], res.params[1], res.params[2]]
    s4 = par[0] + par[1] * window[WE - 1] ** 2 + par[
        2] * res.conditional_volatility[-1] ** 2
    VaR[t, 3] = -np.sqrt(s4) * stats.norm.ppf(p) * value  # GARCH(1,1)

# Backtesting analysis in Python
W1 = WE # Python index starts at 0
m = ["EWMA", "MA", "HS", "GARCH"]

for i in range(4):
    VR = sum(y[W1:T] < -VaR[W1:T,i])/(p*(T-WE))
    s = np.std(VaR[W1:T, i], ddof=1)
    print ([i, m[i], VR, s])

plt.plot(y[W1:T])
plt.plot(VaR[W1:T])
plt.show()
plt.close()
