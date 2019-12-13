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

num_simulations = 10000
time_scaler = 1
calc_type= 'log'
period_interval = 252
confidence_interval = .95
price_col = 'Adj Close'
lambda_decay = .98

input_df = data.loc[:, (portfolio_securities, price_col)]
input_df.columns = input_df.columns.droplevel(1)
input_df = input_df.dropna(axis=1, how='all').dropna(axis=0, how='any')


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
d = ValueAtRisk(interval=confidence_interval,
                matrix=input_df,
                weights=portfolio_securities_weights,
                return_method=calc_type,
                lookbackWindow=period_interval,
                timeScaler=time_scaler,
                sma=True,
                lambda_decay=lambda_decay)

d = MonteCarloVaR(interval=confidence_interval,
                  matrix=input_df,
                  weights=portfolio_securities_weights,
                  return_method=calc_type,
                  lookbackWindow=period_interval,
                  numSimulations=num_simulations)