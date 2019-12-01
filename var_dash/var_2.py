from constants import *
from datetime import datetime
from functions import *
from textwrap import dedent as d

data = pd.read_pickle(hist_pkl)
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
style_markdown = {'fontSize': 16, 'margin-left': 'auto', 'padding': '2px 5px'}
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


def get_returns(securities, calc_type, price_col):
    dff = df.iloc[:, data.columns.get_level_values(1) == price_col]
    dff.columns = dff.columns.droplevel(1)
    dff = dff[securities]
    dff = dff.dropna(axis=1, how='all')
    dff = dff.dropna(axis=0, how='any')
    if calc_type == 'Percentage':
        returns = dff.pct_change().iloc[1:]
        returns.sort_index()
        return returns
    if calc_type == 'Log':
        returns = np.log(dff) - np.log(dff.shift(1))
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
        html.Hr(), dcc.Markdown(d(""" **Return Calculation Method** """), style=style_markdown),
        dcc.RadioItems(
            id='VaR-return-calculation-type',
            options=[{'label': i, 'value': i} for i in ['Percentage', 'Log']],
            value='Percentage',
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
            y=data.values*100,
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


@app.callback(
    # [Output(component_id='VaR-graph', component_property='figure'),
    #  Output(component_id='VaR-result', component_property='children')],
    Output(component_id='body-div', component_property='children'),
    [Input(component_id='show-VaR', component_property='n_clicks')],
    [
        State(component_id='symbol-dropdown', component_property='value'),
        State(component_id='VaR-return-calculation-type', component_property='value'),
        State(component_id='VaR-price-column', component_property='value'),
        State(component_id='VaR-period', component_property='value'),
        State(component_id='VaR-confidence', component_property='value'),
        State(component_id='VaR-series-option', component_property='value')
    ])
def calculateVar(n_clicks, securities, calc_type, price_col, period_interval, confidence_interval, series):
    if n_clicks is None:
        raise PreventUpdate
    else:
        weights = np.repeat(1 / len(securities), len(securities))
        returns = get_returns(securities, calc_type, price_col)
        returns['portfolio_return'] = returns.dot(weights)
        if series == 'False':
            Value_at_Risk = -np.percentile(returns[-period_interval:], 1 - confidence_interval)
            _ = 'Value at Risk of Portfolio = {}'.format(Value_at_Risk)
            h = html.P(children=_, id='VaR-result', style={'color': 'red'})
            g = get_var_graph(returns, period_interval)
            return [g, h]
        if series == 'True':
            print('returns')
            print(returns)
            var_series = get_var_series(returns, period_interval, confidence_interval)
            var_df = var_series[period_interval:].to_frame()
            var_df['time'] = var_df.index
            var_df = var_df[['time', 'var_series']]
            v = generate_table(var_df, max_rows=10)
            g = get_var_portfolio(returns['portfolio_return'], var_series)
            return [g, v]


if __name__ == '__main__':
    app.run_server(debug=True)
