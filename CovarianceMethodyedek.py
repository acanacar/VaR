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

    return VaR_value, portfolio_mean, portfolio_std, cov_mat


df, daily_returns = get_df(data=data, col='Adj Close')
securities, weights = get_portfolio(data=df)
weights = np.array(weights)
actual_portfolio_return = daily_returns[securities].dot(weights)
actual_portfolio_return.name = 'actual_return'

# multiple periods,ci
period_intervals = [252]
confidence_intervals = [.997]

dfVaRs = []
dfVaRs.append(
    (actual_portfolio_return, {'securities': securities,
                               'period_interval': 'return',
                               'confidence_interval': 'return'
                               }))

for p_i_ in period_intervals:
    for c_i_ in confidence_intervals:
        df_VaR, portfolio_mean, portfolio_std, covariance_matrix = VaR_Covariance(
            returns_=daily_returns,
            weights=weights, securities=securities,
            period=p_i_, confidence=c_i_, Series=True
        )
        title = {'securities': securities,
                 'periodic_interval': p_i_,
                 'confidence_interval': c_i_}
        dfVaRs.append((df_VaR, title))
        print(title, ' is done')

for show_ci in confidence_intervals:
    All = pd.concat([df for df, title in dfVaRs if
                     title['confidence_interval'] == show_ci or title['confidence_interval'] == 'return'],
                    axis=1)
    All = All * 100
    All['actual_return'] = All['actual_return'] * -1
    plt.title(str(securities) + str(weights))
    All.plot(lw=1)
    png_name = '/CovarianceMethod_{}.png'.format(show_ci)
    png_path = VaR_png_output_path + png_name
    plt.savefig(png_path)

'''single period,confidence
dfVaR = VaR_Covariance(
    returns_=daily_returns,
    weights=weights, securities=securities,
    period=252, confidence=.95, Series=True
)
print(list(zip(securities, weights)))

print(dfVaR)
'''

import seaborn as sns

sns.set(style='white')

ccov = daily_returns[securities].cov() * 252

# Generate a mask for the upper triangle
mask = np.zeros_like(ccov, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(ccov, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
