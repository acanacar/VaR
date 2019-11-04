import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from scipy.stats import norm
import yfinance as yf
from tabulate import tabulate

df = yf.download('FB', '2012-01-01', '2018-01-31')

df = df[['Close']]
df['returns'] = df.Close.pct_change()

# Variance-Covariance approach
mean = np.mean(df['returns'])
std_dev = np.std(df['returns'])

df['returns'].hist(bins=40, normed=True, histtype='stepfilled', alpha=0.5)

x = np.linspace(mean - 3 * std_dev, mean + 3 * std_dev, 100)
plt.plot(x, norm.pdf(x, mean, std_dev), 'r')
plt.show()

VaR_90 = norm.ppf(1 - 0.9, mean, std_dev)
VaR_95 = norm.ppf(1 - 0.95, mean, std_dev)
VaR_99 = norm.ppf(1 - 0.99, mean, std_dev)

print(tabulate([['90%', VaR_90], ['95%', VaR_95], ['99%', VaR_99]], headers=['Confidence Level', 'Value at Risk']))

# Historical Simulation approach

df = yf.download('FB', '2012-01-01', '2018-01-31')
df['returns'] = df.Close.pct_change()
df = df.dropna()
plt.hist(df.returns, bins=40)
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

df.sort_values('returns', inplace=True, ascending=True)

VaR_90 = df['returns'].quantile(.1)
VaR_95 = df['returns'].quantile(.05)
VaR_99 = df['returns'].quantile(.01)

print(tabulate([['90%', VaR_90], ['95%', VaR_95], ['99%', VaR_99]], headers=['Confidence Level', 'Value at Risk']))
