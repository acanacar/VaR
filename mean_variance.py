from constants import *
from functions import *
from scipy.stats import norm

store = pd.HDFStore(hist_store)

data = store['/all']
data.columns = data.columns.swaplevel(0, 1)


def VaR_Covariance(returns_, period, confidence, weights, securities, Series=False):
    returns_ = returns_[securities]
    if Series == False:
        returns = returns_.iloc[-period:]

        means = returns.mean()
        cov_mat = returns.cov() * period
        # calculate portfolio stdev
        portfolio_var = np.dot(weights.T, np.dot(cov_mat, weights))
        portfolio_std = portfolio_var ** .5
        #calculate portfolio mean
        portfolio_mean = np.dot(weights, means)

        VaR_value = norm.ppf(confidence, portfolio_mean, portfolio_std)
    if Series == True:
        series_name = 'CovarianceMethod_{}_{}'.format(period, confidence)
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


df, daily_returns = get_df(data=data, col='Adj Close')
securities, weights = get_portfolio(data=df)
weights = np.array(weights)

dfVaR = VaR_Covariance(
    returns_=daily_returns,
    weights=weights, securities=securities,
    period=252, confidence=.95,Series=True
)
print(list(zip(securities, weights)))

print(dfVaR)

# period_intervals = [100, 252, 350, 500]
# confidence_intervals = [.68, .95, .997, 1]
