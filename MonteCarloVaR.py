from constants import *
from functions import *
from scipy.stats import norm


def getPortfolioMeanStd(data, period, weights):
    means = data.mean()
    cov_mat = data.cov() * period
    # calculate portfolio stdev
    portfolio_variance = np.dot(weights.T, np.dot(cov_mat, weights))
    portfolio_std = portfolio_variance ** .5
    portfolio_mean = np.dot(weights, means)
    return portfolio_variance, portfolio_std, portfolio_mean


def simulate(data, mean, std, predicted_days=252, num_simulations=10000):
    simulation_df = pd.DataFrame()

    last_price = data[-1]
    for x in range(num_simulations):


        price_series = []
        price_series.append(last_price)
        count = 0

        # Series for Predicted Days
        for i in range(predicted_days):
            if count == predicted_days:
                break

            price = price_series[count] * (1 + np.random.normal(mean, std))

            price_series.append(price)
            count += 1

        simulation_df[x] = price_series

    return simulation_df, last_price


def VaR_MonteCarlo(prices, returns_, period, weights, predicted_days, Series=False):
    if Series == False:
        returns = returns_.iloc[-period:]
        portfolio_variance, portfolio_std, portfolio_mean = getPortfolioMeanStd(data=returns, period=period,
                                                                           weights=weights)
        portfolio_prices = np.dot(prices, weights.T)
        simulation_df, last_price = simulate(data=portfolio_prices,
                                             mean=portfolio_mean,
                                             std=portfolio_std,
                                             predicted_days=predicted_days)

        return simulation_df, last_price


store = pd.HDFStore(hist_store)

data = store['/all']
data.columns = data.columns.swaplevel(0, 1)

df, daily_returns = get_df(data=data, col='Adj Close')
securities, weights = get_portfolio(data=df)
weights = np.array(weights)

df, daily_returns = df[securities], daily_returns[securities]

simulation_df, last_price = VaR_MonteCarlo(df, daily_returns, period=252, weights=weights, predicted_days=1)

price_array = simulation_df.iloc[-1, :]

var68 = np.percentile(price_array, 1 - .68)
var95 = np.percentile(price_array, 1 - .95)
var99 = np.percentile(price_array, 1 - .99)
