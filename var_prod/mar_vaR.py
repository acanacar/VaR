import numpy as np
import pandas as pd
from scipy.stats import norm

Value = 1e6  # $1,000,000
CI = 0.99  # set the confidence interval

tickers = ['AAPL', 'MSFT', 'GOOG']
numbers = len(tickers)

import yfinance as yf

data = pd.DataFrame()
for share in tickers:
    data[share] = yf.download(share, '2014-01-01', '2018-01-31')['Adj Close']
    # data[share]=web.DataReader(share, data_source='yahoo', start='2011-01-01', end='2015-05-15')['Adj Close']
data.columns = tickers

ret = data / data.shift(1) - 1  # calculate the simple returns
ret.mean() * 252  # annualize the returns
covariances = ret.cov() * 252  # gives the annualized covariance of returns
variances = np.diag(covariances)  # extracts variances of the individual shares from covariance matrix
volatility = np.sqrt(variances)  # gives standard deviation

weights = np.random.random(numbers)
weights /= np.sum(
    weights)  # simulating random percentage of exposure of each share that sum up to 1; if we want to plug in our own weights use: weights=np.array([xx,xx,xx])

Pf_ret = np.sum(ret.mean() * weights) * 252  # Portfolio return

Pf_variance = np.dot(weights.T, np.dot(ret.cov() * 252, weights))  # Portfolio variance
Pf_volatility = np.sqrt(Pf_variance)  # Portfolio standard deviation

USDvariance = np.square(Value) * Pf_variance
USDvolatility = np.sqrt(USDvariance)

covariance_asset_portfolio = np.dot(weights.T, covariances)
covUSD = np.multiply(covariance_asset_portfolio, Value)
beta = np.divide(covariance_asset_portfolio, Pf_variance)


def VaR():
    # this code calculates Portfolio Value-at-risk (Pf_VaR) in USD-terms and Individual Value-at-risk (IndividualVaR) of shares in portfolio.
    Pf_VaR = norm.ppf(CI) * USDvolatility
    IndividualVaR = np.multiply(volatility, Value * weights) * norm.ppf(CI)
    IndividualVaR = ['$%.2f' % elem for elem in IndividualVaR]
    print('Portfolio VaR: ', '$%0.2f' % Pf_VaR)
    print('Individual VaR: ',
          [[tickers[i], IndividualVaR[i]] for i in range(min(len(tickers), len(IndividualVaR)))])


VaR()  # call the function to get portfolio VaR and Individual VaR in USD-terms


def marginal_component_VaR():
    # this code calculates Marginal Value-at-risk in USD-terms and Component Value-at-risk of shares in portfolio.
    marginalVaR = np.divide(covUSD, USDvolatility) * norm.ppf(CI)
    componentVaR = np.multiply(weights, beta) * USDvolatility * norm.ppf(CI)
    marginalVaR = ['%.3f' % elem for elem in marginalVaR]
    componentVaR = ['$%.2f' % elem for elem in componentVaR]
    print('Marginal VaR:', [[tickers[i], marginalVaR[i]] for i in range(min(len(tickers), len(marginalVaR)))])
    print('Component VaR: ',
          [[tickers[i], componentVaR[i]] for i in range(min(len(tickers), len(componentVaR)))])


marginal_component_VaR()
