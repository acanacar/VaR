from constants import *
from functions import *
from scipy.stats import norm
import matplotlib.mlab as mlab

store = pd.HDFStore(hist_store)

data = store['/all']
data.columns = data.columns.swaplevel(0, 1)

log = 0
lambda_decay = .9
return_change = 'log' # 'pct'
period = 252
time_scaler = 1
confidence = .99
check_losses_at = 1 - confidence

df, daily_returns = get_df(data=data, col='Adj Close',change=return_change)  # pct change
securities, weights = get_portfolio(data=df)
weights = np.array(weights)

daily_returns = daily_returns[securities]
portfolio_return_series = daily_returns.dot(weights)
portfolio_return_series_square = portfolio_return_series ** 2


def graph_VaR(std, ret, value_var,date):
    x = np.linspace(ret - 4 * std, \
                    ret + 4 * std, 100)
    plt.plot(x, mlab.normpdf(x, ret, std))
    weights_ = list(map(lambda x:'{:.0%}'.format(x),weights))
    sec_weights_title = dict(zip(securities, weights_))
    sec_weights_title = str(sec_weights_title).replace('{', '').replace('}', '').replace("'", '')
    plt.title("VaR {}\n{}".format(date,sec_weights_title))
    plt.xlabel("Portfolio Return")
    # Plot the normal curve and label the x axis and the graph

    lower_limit = ret - 4 * std
    upper_limit = ret + 4 * std
    increment = (upper_limit - lower_limit) / 150

    height_at_critical = norm(ret, std).pdf(lower_limit)
    plt.plot((lower_limit, lower_limit), (0, height_at_critical), 'r')

    #
    xStart = lower_limit
    xEnd = value_var
    yEnd = norm(ret, std).pdf(value_var)
    yStart = yEnd + 3
    plt.annotate('{:6.4f}'.format(value_var),
                 xytext=(xStart, yStart),
                 xy=(xEnd, yEnd),
                 arrowprops=dict(facecolor='red', shrink=0.01))
    #
    for i in range(150):
        lower_limit += increment
        height_at_critical = norm(ret, std).pdf(lower_limit)
        if lower_limit < value_var:
            plt.plot((lower_limit, lower_limit), (0, height_at_critical), 'r')
        else:
            plt.plot((lower_limit, lower_limit), (0, height_at_critical), 'b', alpha=.4)
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
    returns['portfolio_return'] = returns.dot(weights)
    portfolio_mean = returns['portfolio_return'].mean()

    Range = list(range(period))
    Range.reverse()
    returns['Range'] = Range
    returns['weights'] = returns.apply(
        lambda row_: (1 - lambda_decay) * (lambda_decay ** row_['Range']), axis=1)
    returns['scaled_weights'] = returns['weights'] / (1 - (lambda_decay ** period))

    portfolio_variance = sum([row_.scaled_weights*(row_.portfolio_return - portfolio_mean)**2 for row_ in returns.itertuples()])
    portfolio_std_dev = portfolio_variance**.5

    '''# burasinin uzerine dusunulebilir portfolio meani olarak 0 mi almaliyiz populasyon gibi dusunerek'''
    # var = norm(0, portfolio_variance ** .5).ppf(check_losses_at)
    var = norm(portfolio_mean, portfolio_variance ** .5).ppf(check_losses_at)

    var_series[-i - 1] = var
    mean_series[-i - 1] = portfolio_mean
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
              value_var=row.portfolio_VaR,
              date=row.name.strftime('%d-%m-%Y'))

p.show()

#
backtest_df = daily_returns[['portfolio_VaR', 'next_day_return']] * -1
ax = backtest_df.plot(lw=1)
ax.set_ylim(0, backtest_df['next_day_return'].nlargest(2)[-1])
plt.show()
#
last_year_failure_amt = daily_returns['VaR_fail_flag'].rolling(window=252).sum()
print('son 1 yildaki max fail miktari: ', int(last_year_failure_amt.max()))


#
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
#