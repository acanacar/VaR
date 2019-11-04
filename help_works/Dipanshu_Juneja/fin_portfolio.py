# https://www.youtube.com/watch?v=4hy8Am4EIBo  Financial Portfolio Analysis with Python


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quandl

Mr_Techie = ['MSFT', 'NFLX', 'FB', 'AMZN']
Mr_Allstar = ['MSFT', 'PFE', 'F', 'WMT']

techie_portfolio = pd.DataFrame()
allstar_portfolio = pd.DataFrame()

for tech, allstar in zip(Mr_Techie, Mr_Allstar):
    techie_portfolio[tech] = yf.download(tech,'2013-01-01','2018-01-31')['Adj Close']
    allstar_portfolio[allstar] = yf.download(allstar,'2013-01-01','2018-01-31')['Adj Close']
    # techie_portfolio[tech] = quandl.get("EOD/{}".format(tech), authtoken="CBf_Ha_rJsGWkWNz1BU8")['Adj_Close']
    # allstar_portfolio[allstar] = quandl.get("EOD/{}".format(allstar), authtoken="CBf_Ha_rJsGWkWNz1BU8")['Adj_Close']


(techie_portfolio / techie_portfolio.iloc[0]).plot(figsize=(15, 6))
plt.show()
(allstar_portfolio / allstar_portfolio.iloc[0]).plot(figsize=(15, 6))
plt.show()

techie_portfolio.shape

techie_avg_returns = (pow(techie_portfolio.iloc[-1] / techie_portfolio.iloc[0], 1 / 1234) - 1) * 250
allstar_avg_returns = (pow(allstar_portfolio.iloc[-1] / allstar_portfolio.iloc[0], 1 / 1234) - 1) * 250
print(techie_avg_returns)
print(allstar_avg_returns)

weights = np.array([0.25, 0.25, 0.25, 0.25])
techie_portfolio_return = np.dot(weights, techie_avg_returns)
allstar_portfolio_return = np.dot(weights, allstar_avg_returns)

print(techie_portfolio_return)
print(allstar_portfolio_return)

techie_daily_returns = ((techie_portfolio / techie_portfolio.shift(-1)) - 1)
allstar_daily_returns = ((allstar_portfolio / allstar_portfolio.shift(-1)) - 1)

print(techie_daily_returns.head())
print(allstar_daily_returns.head())

# Portfolio Risk
covar_techie = techie_daily_returns.cov() * 250
covar_allstar = allstar_daily_returns.cov() * 250

print(covar_techie)
print(covar_allstar)

techie_portfolio_var = np.dot(weights.T, np.dot(covar_techie, weights))
allstar_portfolio_var = np.dot(weights.T, np.dot(covar_allstar, weights))

techie_portfolio_risk = techie_portfolio_var ** 0.5
allstar_portfolio_risk = allstar_portfolio_var ** 0.5

# Sharpe Ratio

rf = 0.02

sharpeRatioTechie = (techie_portfolio_return - rf) / techie_portfolio_risk
sharpeRatioAllstar = (allstar_portfolio_return - rf) / allstar_portfolio_risk

print('Sharpe ratio of Mr techie ', sharpeRatioTechie)
print('Sharpe ratio of Mr Allstar ', sharpeRatioAllstar)
