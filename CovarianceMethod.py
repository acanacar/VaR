from constants import *
from functions import *
from scipy.stats import norm

store = pd.HDFStore(hist_store)

data = store['/all']
data.columns = data.columns.swaplevel(0, 1)

log = 0
period = 252
time_scaler = 1
confidence = .99
check_losses_at = 1 - confidence

df, daily_returns = get_df(data=data, col='Adj Close')  # pct change
securities, weights = get_portfolio(data=df)
weights = np.array(weights)

daily_returns = daily_returns[securities]
portfolio_return_series = daily_returns.dot(weights)


def getValueAtRisk(data):
    mean_return = data.mean()
    covariance_matrix = data.cov()
    portfolio_return = sum(mean_return * weights) * time_scaler
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights))) * np.sqrt(time_scaler)

    potential_loss = norm(portfolio_return, portfolio_std_dev).ppf(check_losses_at)
    return (potential_loss, covariance_matrix, portfolio_return, portfolio_std_dev)


import matplotlib.mlab as mlab


def graph_VaR(std, ret, value_var):
    x = np.linspace(portfolio_return - 4 * portfolio_std_dev, \
                    portfolio_return + 4 * portfolio_std_dev, 100)
    plt.plot(x, mlab.normpdf(x, portfolio_return, portfolio_std_dev))
    plt.title("Value At Risk")
    plt.xlabel("Portfolio Return")
    # Plot the normal curve and label the x axis and the graph

    lower_limit = portfolio_return - 4 * portfolio_std_dev
    upper_limit = portfolio_return + 4 * portfolio_std_dev
    increment = (upper_limit - lower_limit) / 150
    # Establish the lower and upper limit of our normal curve that we plan to plot
    # Initialize an increment value for countless intervals inbetween

    height_at_critical = norm(portfolio_return, portfolio_std_dev).pdf(lower_limit)
    plt.plot((lower_limit, lower_limit), (0, height_at_critical), 'r')

    for i in range(150):
        lower_limit += increment
        height_at_critical = norm(portfolio_return, portfolio_std_dev).pdf(lower_limit)
        if lower_limit < value_var:
            plt.plot((lower_limit, lower_limit), (0, height_at_critical), 'r')
        else:
            plt.plot((lower_limit, lower_limit), (0, height_at_critical), 'b')
    return plt


var_series = pd.Series(index=daily_returns.index, name='var_series')
mean_series = pd.Series(index=daily_returns.index, name='mean_series')
# covariance_matrix_series = pd.Series(index=daily_returns.index, name='cov_mat_series')
std_series = pd.Series(index=daily_returns.index, name='std_series')

for i in range(len(daily_returns) - period):
    if i == 0:
        # at first in loop
        returns = daily_returns.iloc[-period:]
    else:
        returns = daily_returns.iloc[-(period + i):-i]

    (var, covariance_matrix, portfolio_return, portfolio_std_dev) = getValueAtRisk(data=returns)
    # response = getValueAtRisk(data=returns)
    # var = response[0]

    var_series[-i - 1] = var
    mean_series[-i - 1] = portfolio_return
    # covariance_matrix_series[-i - 1] = covariance_matrix
    std_series[-i - 1] = portfolio_std_dev

    if log == 1:
        print("At the " + str(confidence * 100) + " percent level of confidence,\
         the portfolio may experience a return of " + "{0:0.1f}".format(var * 100) + \
              " or less.")

daily_returns['portfolio_VaR'] = var_series
daily_returns['portfolio_std'] = std_series
daily_returns['portfolio_mean'] = mean_series
daily_returns['next_day_return'] = daily_returns[securities].dot(weights).shift(-1)

daily_returns['VaR_fail_flag'] = daily_returns.apply(
    lambda row_: 1 if row_['next_day_return'] < row_['portfolio_VaR'] else 0, axis=1)

n = 400
row = daily_returns.iloc[n]
p = graph_VaR(std=row.portfolio_std,
              ret=row.portfolio_mean,
              value_var=row.portfolio_VaR)
p.show()

backtest_df = daily_returns[['portfolio_VaR', 'next_day_return']] * -1
ax = backtest_df.plot(lw=1)
ax.set_ylim(0, backtest_df['next_day_return'].nlargest(2)[-1])
ax.show()
last_year_failure_amt = daily_returns['VaR_fail_flag'].rolling(window=252).sum()
print('son 1 yildaki max fail miktari: ', int(last_year_failure_amt.max()))


def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins) + bin_w / 2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])


bins = range(last_year_failure_amt.nunique() + 1)

plt.hist(
    last_year_failure_amt,
    bins=bins
)
plt.title('last 1 year model fail amount durations (days)')
bins_labels(bins, fontsize=20)
plt.show()
plt.close()
