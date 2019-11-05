from constants import *
from functions import get_df, get_portfolio,VaR as VaRCalculate


def VaR_Covariance(returns_, period, confidence, weights, securities, Series=False):
    returns_ = returns_[securities]
    if Series == False:
        returns = returns_.iloc[-period:]

        means = returns.mean()
        cov_mat = returns.cov() * period
        # calculate portfolio stdev
        portfolio_var = np.dot(weights.T, np.dot(cov_mat, weights))
        portfolio_std = portfolio_var ** .5
        # calculate portfolio mean
        portfolio_mean = np.dot(weights, means)

        VaR_value = norm.ppf(confidence, portfolio_mean, portfolio_std)
    if Series == True:
        series_name = '{}_{}'.format(period, confidence)
        VaR_value = pd.Series(index=returns_.index, name=series_name)
        for i in range(0, len(returns_) - period):
            if i == 0:
                returns = returns_.iloc[-period:]
            else:
                returns = returns_.iloc[-(period + i):-i]
            means = returns.mean()
            cov_mat = returns.cov() * period
            portfolio_var = np.dot(weights.T, np.dot(cov_mat, weights))
            portfolio_std = portfolio_var ** .5
            portfolio_mean = np.dot(weights, means)

            VaR_value[-i - 1] = norm.ppf(confidence, portfolio_mean, portfolio_std)

    return VaR_value

store = pd.HDFStore(hist_store)

data = store['/all']
data.columns = data.columns.swaplevel(0, 1)

df, daily_returnsx = get_df(data=data, col='Adj Close')

arrays = [
    ['Historical', 'Covariance'],
    ['p_100', 'p_252', 'p_350', 'p_500'],
    ['.68', '.95', '.997', '1']
]
multi_index = pd.MultiIndex.from_product(arrays, names=['method', 'period_interval', 'confidence_interval'])

dx = pd.DataFrame(index=daily_returnsx.index, columns=multi_index)

securities, weights = get_portfolio(data=df)
weights = np.array(weights)
print(securities, '\n', weights)
for col_name, col_data in dx.iteritems():
    method, p_i_, c_i_ = col_name[0], int(col_name[1].split('_')[-1]), float(col_name[2])
    if col_name[0] == 'Covariance':
        df_VaR = VaR_Covariance(
            returns_=daily_returnsx,
            weights=weights, securities=securities,
            period=p_i_, confidence=c_i_, Series=True
        )
        dx.loc[:, col_name] = df_VaR.values

    if col_name[0] == 'Historical':
        df_VaR = VaRCalculate(Data=df,
                     Returns=daily_returnsx,
                     Method=method_lookup[method],
                     Confidence_Interval=c_i_,
                     Period_Interval=p_i_,
                     Series=True)
    print(col_name)
    print(len(col_data))

print('x')
