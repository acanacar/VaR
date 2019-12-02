from constants import *
from datetime import datetime
from functions import *
from textwrap import dedent as d
from var_class.var_1 import *

data = pd.read_pickle(hist_pkl)
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
style_markdown = {'fontSize': 16, 'margin-left': 'auto', 'padding': '2px 5px'}
var_methods = [
    ('Basic Historical Simulation', 'Basic-Historical-Simulation'),
    ('Age Weighted Historical Simulation', 'Age-Weighted-Historical-Simulation'),
    ('Parametric', 'Parametric'),
    ('Monte Carlo Simulation', 'Monte-Carlo-Simulation'),
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
df = pd.read_pickle(hist_pkl)


def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [
            html.Tr(
                [
                    html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
                ]
            ) for i in range(min(len(dataframe), max_rows))
        ]
    )


def get_returns(data, calc_type):
    if calc_type == 'pct':
        returns = data.pct_change().iloc[1:]
        returns.sort_index()
        return returns
    if calc_type == 'log':
        returns = np.log(data) - np.log(data.shift(1))
        returns.sort_index()
        return returns


app.layout = html.Div([

    html.Div([
        html.Div([
            dcc.Markdown(d(""" **Portfolio Stocks** """), style=style_markdown),
            dcc.Dropdown(
                id='symbol-dropdown',
                options=[{'label': s, 'value': s} for s in data.columns.levels[0].values],
                value=['AKBNK.IS', 'GARAN.IS'],
                multi=True)], style={'width': '100%'}),
        html.Hr(),
        html.Div([
            dcc.Markdown(d(""" **Value at Risk Method** """), style=style_markdown),
            dcc.Dropdown(
                id='VaR-method-dropdown',
                options=[{'label': l, 'value': v} for l, v in var_methods],
                value='Basic-Historical-Simulation'
            )], style={'width': '50%'}),
        html.Hr(), dcc.Markdown(d(""" **Return Calculation Method** """), style=style_markdown),
        dcc.RadioItems(
            id='VaR-return-calculation-type',
            options=[{'label': l, 'value': v} for l, v in [('Percentage', 'pct'), ('Log', 'log')]],
            value='pct',
            labelStyle={'display': 'inline-block'}
        ),
        html.Hr(), dcc.Markdown(d(""" **Price Column** """), style=style_markdown),
        dcc.RadioItems(
            id='VaR-price-column',
            options=[{'label': i, 'value': i} for i in ['Close', 'Adj Close']],
            value='Adj Close',
            labelStyle={'display': 'inline-block'}
        ),
        html.Hr(), dcc.Markdown(d("""
            **Period Interval**

            """), style=style_markdown),
        dcc.Input(
            id='VaR-period',
            type='number',
            placeholder='period interval',
            value=252
        ),
        html.Hr(), dcc.Markdown(d(""" **Confidence Interval** """), style=style_markdown),
        dcc.RadioItems(
            id='VaR-confidence',
            options=[
                {'label': '68%', 'value': .68},
                {'label': '95%', 'value': .95},
                {'label': '99%', 'value': .99},
            ],
            value=.99,
            labelStyle={'display': 'inline-block'},
        ),
        html.Hr(), dcc.Markdown(d(""" **Series T/F** """), style=style_markdown),
        dcc.RadioItems(
            id='VaR-series-option',
            options=[{'label': i, 'value': i} for i in ['True', 'False']],
            value='False',
            labelStyle={'display': 'inline-block'},
        ),
        html.Button('Calculate VaR', id='show-VaR'),

    ], style={'width': '33%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div(id='body-div',
             # children=[
             #     dcc.Graph(id='VaR-graph'), html.P(id='VaR-result', style={'color': 'red'})],
             style={'width': '60%', 'display': 'inline-block', 'padding': '0 20'})

])


def get_var_graph(returns, period_interval):
    returns = returns.iloc[-period_interval:]
    traces = []
    for col in returns.columns:
        traces.append(dict(
            x=returns.index,
            y=returns[col],
            mode='lines',
            name=col,
            opacity=0.7
        ))
    return dcc.Graph(
        id='stocks-graph', figure={
            'data': traces,
            'layout': {
                'height': 500,
                'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
                'xaxis': {'showgrid': False}
            }
        })


def get_var_portfolio(portfolio_returns, var_series):
    traces = []
    for data in [portfolio_returns, var_series]:
        traces.append(dict(
            x=data.index,
            y=data.values * 100,
            mode='lines',
            name=data.name
        ))
    return dcc.Graph(
        id='stocks-graph', figure={
            'data': traces,
            'layout': {
                'height': 500,
                'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
                'xaxis': {'showgrid': False}
            }
        })


def get_var_series(returns, period_interval, confidence_interval):
    Value_at_Risk = pd.Series(index=returns.index, name='var_series')
    for i in range(0, len(returns) - period_interval):
        if i == 0:
            Data = returns[-period_interval:]  # alttaki satirin nin daha sade matematiksel sekli
            # Data = Returns[-Period_Interval+i:]
        else:
            Data = returns[-(period_interval + i):-i]
        Value_at_Risk[-i - 1] = -np.percentile(Data, 1 - confidence_interval)
    return Value_at_Risk


def get_weights(n):
    lis = np.random.rand(n)
    lis_sum = functools.reduce(lambda a, b: a + b, lis)
    portfolio_securities_weights = list(map(lambda y: y / lis_sum, lis))
    return portfolio_securities_weights


def get_input_df(data, portfolio_securities, price_col):
    df = data.loc[:, (portfolio_securities, price_col)]
    df.columns = df.columns.droplevel(1)
    df = df.dropna(axis=1, how='all').dropna(axis=0, how='any')
    return df


def get_vaR_instance(input_df, weights, method, calc_type, period_interval, confidence_interval):
    if method == 'Basic-Historical-Simulation':

        d = HistoricalVaR(interval=confidence_interval,
                          matrix=input_df,
                          weights=weights,
                          return_method=calc_type,
                          lookbackWindow=period_interval
                          )
    elif method == 'Age-Weighted-Historical-Simulation':
        d = HistoricalVaR(interval=confidence_interval,
                          matrix=input_df,
                          weights=weights,
                          return_method=calc_type,
                          lookbackWindow=period_interval,
                          hybrid=True
                          )
    elif method == 'Parametric':
        d = ValueAtRisk(interval=confidence_interval,
                        matrix=input_df,
                        weights=weights,
                        return_method=calc_type,
                        lookbackWindow=period_interval)
    elif method == 'Monte-Carlo-Simulation':
        d = MonteCarloVaR(interval=confidence_interval,
                          matrix=input_df,
                          weights=weights,
                          return_method=calc_type,
                          lookbackWindow=period_interval)
    else:
        print('unvalid method')
    return d


@app.callback(
    # [Output(component_id='VaR-graph', component_property='figure'),
    #  Output(component_id='VaR-result', component_property='children')],
    Output(component_id='body-div', component_property='children'),
    [Input(component_id='show-VaR', component_property='n_clicks')],
    [
        State(component_id='VaR-method-dropdown', component_property='value'),
        State(component_id='symbol-dropdown', component_property='value'),
        State(component_id='VaR-return-calculation-type', component_property='value'),
        State(component_id='VaR-price-column', component_property='value'),
        State(component_id='VaR-period', component_property='value'),
        State(component_id='VaR-confidence', component_property='value'),
        State(component_id='VaR-series-option', component_property='value')
    ])
def calculateVar(n_clicks, method, securities, calc_type, price_col, period_interval, confidence_interval,
                 series):
    if n_clicks is None:
        raise PreventUpdate
    else:
        input_df = get_input_df(data=data, portfolio_securities=securities, price_col=price_col)
        weights = get_weights(n=len(securities))
        d = get_vaR_instance(input_df, weights, method,
                             calc_type, period_interval, confidence_interval)
        returns = get_returns(input_df, calc_type)
        returns['portfolio_return'] = returns.dot(weights)
        if series == 'False':
            Value_at_Risk = d.vaR()
            _ = 'Value at Risk of Portfolio = {}'.format(Value_at_Risk)
            h = html.P(children=_, id='VaR-result', style={'color': 'red'})
            g = get_var_graph(returns, period_interval)
            return [g, h]
        if series == 'True':
            var_series = d.vaR(series=True)
            var_df = var_series[period_interval:].to_frame()
            var_df['time'] = var_df.index
            var_df = var_df[['time', 'var_series']]
            v = generate_table(var_df, max_rows=10)
            g = get_var_portfolio(returns['portfolio_return'], var_series)
            return [g, v]


if __name__ == '__main__':
    app.run_server(debug=True)
