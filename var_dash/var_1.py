from constants import *
from datetime import datetime
from functions import *

data = pd.read_pickle(hist_pkl)
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
style_1 = {'fontSize': 12, 'margin-left': 'auto', 'padding': '3px'}
style_div_1 = {'display': 'inline-block', 'margin-top': '10px'}
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def VaR(Data, Returns, Method='Historical_Simulation', Confidence_Interval=0.95, Period_Interval=None,
        Series=False):
    print('vaR xxx')
    if Method == 'Historical_Simulation':

        if Series == False:
            Data = Returns[-Period_Interval:]
            Value_at_Risk = -np.percentile(Data, 1 - Confidence_Interval)
        if Series == True:
            series_name = '{}_{}_{}'.format(Method, Period_Interval, Confidence_Interval)
            Value_at_Risk = pd.Series(index=Returns.index, name=series_name)
            for i in range(0, len(Returns) - Period_Interval):
                if i == 0:
                    Data = Returns[-Period_Interval:]  # alttaki satirin nin daha sade matematiksel sekli
                    # Data = Returns[-Period_Interval+i:]
                else:
                    Data = Returns[-(Period_Interval + i):-i]
                Value_at_Risk[-i - 1] = -np.percentile(Data, 1 - Confidence_Interval)

    return (Value_at_Risk)


def run_calc(stocks, price_type, calc_type, confidence_interval, period_interval):
    print('xxx')
    weights = np.repeat(1 / len(stocks), len(stocks))
    df = data.iloc[:, data.columns.get_level_values(1) == price_type]
    df.columns = df.columns.swaplevel(0, 1)
    df.columns = df.columns.droplevel()
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, how='any')
    if calc_type == 'pct':
        returns = df.pct_change().iloc[1:]
    if calc_type == 'log':
        returns = np.log(df) - np.log(df.shift(1))
    returns.sort_index()
    print('ccc')
    print(returns)
    returns['return'] = returns.dot(weights)
    portfolio_return = returns['return'] * 100  # grafikteki daha guzel goruntu icin
    df_VaR = VaR(Data=df,
                 Returns=portfolio_return,
                 Method='Historical_Simulation',
                 Confidence_Interval=confidence_interval,
                 Period_Interval=period_interval,
                 Series=True)
    print(df_VaR)
    return df_VaR


app.layout = html.Div(children=[
    html.Div(children='''
    Symbol to graph
    '''),
    # dcc.Checklist(
    #     id='symbol-dropdown',
    #     options=[{'label': s, 'value': s} for s in data.columns.levels[0].values],
    #     value=['AKBNK.IS']
    # ),
    # html.Div([html.H3('Enter start / end date:'),
    #           dcc.DatePickerRange(id='my_date_picker',
    #                               min_date_allowed=datetime(2015, 1, 1),
    #                               max_date_allowed=datetime.today(),
    #                               start_date=datetime(2018, 1, 1),
    #                               end_date=datetime.today()
    #                               )
    #
    #           ], style={'display': 'inline-block'}),
    html.Div([
        dcc.Dropdown(
            id='symbol-dropdown',
            options=[{'label': s, 'value': s} for s in data.columns.levels[0].values],
            value='AKBNK.IS',
            multi=True)], style={'width': '50%'}),
    # adj close vs close
    html.Div([
        dcc.Dropdown(
            id='price-type-dropdown',
            options=[{'label': 'Close', 'value': 'Close'},
                     {'label': 'Adj Close', 'value': 'Adj Close'}],
            value='Adj Close',
            style=style_1),
        # pct vs log
        dcc.RadioItems(
            id='return_calculation_type',
            options=[
                {'label': 'Logarithmic Change', 'value': 'log'},
                {'label': 'Percentage Change', 'value': 'pct'},
            ],
            value='pct',
            labelStyle={'display': 'inline-block'},
            style=style_1
        ),
        # period
        dcc.Input(
            id='input-period',
            type='number',
            placeholder='period interval',
        ),
        # confidence_interval
        dcc.RadioItems(
            id='input-confidence',
            options=[
                {'label': '68%', 'value': .68},
                {'label': '95%', 'value': .95},
                {'label': '99%', 'value': .99},
            ],
            value=.99,
            labelStyle={'display': 'inline-block'},
            style=style_1
        ),
        # submit button
        html.Button(children='Submit',
                    id='submit-button',
                    n_clicks=0,
                    style=style_1)
    ], style=style_div_1),

    html.Div(id='output-graph')
])


@app.callback(
    Output(component_id='output-graph', component_property='children'),
    [Input(component_id='submit-button', component_property='n_clicks')],
    [State(component_id='symbol-dropdown', component_property='value'),
     State(component_id='price-type-dropdown', component_property='value'),
     State(component_id='input-confidence', component_property='value'),
     State(component_id='input-period', component_property='value'),
     State(component_id='return_calculation_type', component_property='value')
     ])
def update_value(n_clicks, stock_names, price_type, confidence_interval, period_interval, calc_type):
    # VaR chart
    df = run_calc(stocks=stock_names,
                                    price_type=price_type,
                                    calc_type=calc_type,
                                    confidence_interval=confidence_interval,
                                    period_interval=period_interval)

    return dcc.Graph(
        id='VaR-graph',
        figure={
            'data': [
                {'x': df.index, 'y': df.values, 'type': 'line', 'name': 'portfolio_VaR'}
            ]
        }
    )

    # graphs = []
    # graphs.append(
    #     dcc.Graph(
    #         id='VaR-graph',
    #         figure={
    #             'data': [
    #                 {'x': df_VaR.index, 'y': df_VaR.values, 'type': 'line', 'name': 'portfolio_VaR'}
    #             ]
    #         }
    #     )
    # )
    # # stock chart
    # traces = []
    # for stock_name in stock_names:
    #     df = data[stock_name]
    #     traces.append({'x': df.index, 'y': df[price_type], 'type': 'line', 'name': stock_name})
    #
    # # df = data.iloc[:, data.columns.get_level_values(1) == price_type]
    # fig = {'data': traces, 'layout': {'title': 'Stock Chart'}}
    # graphs.append(dcc.Graph(
    #     id='example-graph',
    #     figure=fig
    # ))

    # return graphs


if __name__ == '__main__':
    app.run_server(debug=True)
