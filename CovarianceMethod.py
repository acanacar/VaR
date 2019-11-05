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


df, daily_returns = get_df(data=data, col='Adj Close')
securities, weights = get_portfolio(data=df)
weights = np.array(weights)
actual_portfolio_return = daily_returns[securities].dot(weights)
actual_portfolio_return.name = 'actual_return'

'''single period,confidence
dfVaR = VaR_Covariance(
    returns_=daily_returns,
    weights=weights, securities=securities,
    period=252, confidence=.95, Series=True
)
print(list(zip(securities, weights)))

print(dfVaR)
'''

# multiple periods,ci
period_intervals = [100, 252, 350, 500]
confidence_intervals = [.68, .95, .997, 1]

dfVaRs = [
    (actual_portfolio_return, {'securities': securities,
                               'period_interval': 'return',
                               'confidence_interval': 'return'
                               })
]
for p_i_ in period_intervals:
    for c_i_ in confidence_intervals:
        df_VaR = VaR_Covariance(
            returns_=daily_returns,
            weights=weights, securities=securities,
            period=p_i_, confidence=c_i_, Series=True
        )
        dfVaRs.append((df_VaR, {'securities': securities,
                                'period_interval': p_i_,
                                'confidence_interval': c_i_}))

for show_ci in confidence_intervals:
    All = pd.concat([df for df, title in dfVaRs if
                     title['confidence_interval'] == show_ci or title['confidence_interval'] == 'return'],
                    axis=1)
    plt.title(str(securities) + str(weights))
    All.plot(lw=1)
    png_name = 'CovarianceMethod_{}'.format(show_ci)
    png_path = r'C:\Users\a.acar\PycharmProjects\VaR\outputs\{}.png'.format(png_name)
    plt.savefig(png_path)
