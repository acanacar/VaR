from constants import *
from functions import *
from scipy.stats import norm
import matplotlib.mlab as mlab

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

daily_returns['portfolio_return'] = daily_returns.dot(weights)
d = {t:sum(daily_returns[t]) for t in daily_returns.columns}

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

print(__version__) # requires version >= 1.9.0
trace1 = go.Bar(
    x=list(d.keys()),
    y=list(d.values()),
    name='Ticker YTD')



data = [trace1]

layout = go.Layout(title='YTD Return vs S&P 500 YTD'
                   , barmode='group'
                   , yaxis=dict(title='Returns', tickformat=".2%")
                   , xaxis=dict(title='Ticker')
                   , legend=dict(x=.8, y=1)
                   )

fig = go.Figure(data=data, layout=layout)
iplot(fig)