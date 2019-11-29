import pandas as pd
from constants import *
from functions import *
from var_class.var_1 import *

data = pd.read_pickle(hist_pkl)

#
all_securities = list(data.columns.get_level_values(0))

#
n = 4
lis = np.random.rand(n)
lis_sum = functools.reduce(lambda a, b: a + b, lis)
portfolio_securities_weights = list(map(lambda y: y / lis_sum, lis))

#
portfolio_securities = random.sample(all_securities, n)

#
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
d_historical = HistoricalVaR(.95, df.as_matrix(), portfolio_securities_weights)
print('VaR(Percentage): ', d_historical.vaR(), '%')
print('Var(Dollar):', d_historical.vaR(marketValue=1000000))
print('100 day - VaR(Percentage): ', d_historical.vaR(window=100) * 100, '%')
print('100 day - Var(Dollar):', d_historical.vaR(marketValue=1000000, window=100))

#
