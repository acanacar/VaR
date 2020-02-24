path = r'C:\Users\a.acar\PycharmProjects\ITCH\test_itch\outputs\Akbnk201805_OB.pkl'
import pandas as pd
from textwrap import dedent as d
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

df = pd.read_pickle(path)

df['time'] = pd.to_datetime(df.index)
df.fillna(method='ffill', inplace=True)

bid_0_p, ask_0_p, bid_0_s, ask_0_s = \
    df.B.map(lambda r: r[0][0] if len(r) > 0 else None, na_action='ignore'), \
    df.S.map(lambda r: r[0][0] if len(r) > 0 else None, na_action='ignore'), \
    df.B.map(lambda r: r[0][1] if len(r) > 0 else None, na_action='ignore'), \
    df.S.map(lambda r: r[0][1] if len(r) > 0 else None, na_action='ignore')
bid_1_p, ask_1_p, bid_1_s, ask_1_s = \
    df.B.map(lambda r: r[1][0] if len(r) > 0 else None, na_action='ignore'), \
    df.S.map(lambda r: r[1][0] if len(r) > 0 else None, na_action='ignore'), \
    df.B.map(lambda r: r[1][1] if len(r) > 0 else None, na_action='ignore'), \
    df.S.map(lambda r: r[1][1] if len(r) > 0 else None, na_action='ignore')
bid_2_p, ask_2_p, bid_2_s, ask_2_s = \
    df.B.map(lambda r: r[2][0] if len(r) > 0 else None, na_action='ignore'), \
    df.S.map(lambda r: r[2][0] if len(r) > 0 else None, na_action='ignore'), \
    df.B.map(lambda r: r[2][1] if len(r) > 0 else None, na_action='ignore'), \
    df.S.map(lambda r: r[2][1] if len(r) > 0 else None, na_action='ignore')
bid_3_p, ask_3_p, bid_3_s, ask_3_s = \
    df.B.map(lambda r: r[3][0] if len(r) > 0 else None, na_action='ignore'), \
    df.S.map(lambda r: r[3][0] if len(r) > 0 else None, na_action='ignore'), \
    df.B.map(lambda r: r[3][1] if len(r) > 0 else None, na_action='ignore'), \
    df.S.map(lambda r: r[3][1] if len(r) > 0 else None, na_action='ignore')
bid_4_p, ask_4_p, bid_4_s, ask_4_s = \
    df.B.map(lambda r: r[4][0] if len(r) > 0 else None, na_action='ignore'), \
    df.S.map(lambda r: r[4][0] if len(r) > 0 else None, na_action='ignore'), \
    df.B.map(lambda r: r[4][1] if len(r) > 0 else None, na_action='ignore'), \
    df.S.map(lambda r: r[4][1] if len(r) > 0 else None, na_action='ignore')

dfx = pd.DataFrame({'bid_0_p': bid_0_p, 'bid_0_s': bid_0_s, 'ask_0_p': ask_0_p, 'ask_0_s': ask_0_s,
                    'bid_1_p': bid_1_p, 'bid_1_s': bid_1_s, 'ask_1_p': ask_1_p, 'ask_1_s': ask_1_s,
                    'bid_2_p': bid_2_p, 'bid_2_s': bid_2_s, 'ask_2_p': ask_2_p, 'ask_2_s': ask_2_s,
                    'bid_3_p': bid_3_p, 'bid_3_s': bid_3_s, 'ask_3_p': ask_3_p, 'ask_3_s': ask_3_s,
                    'bid_4_p': bid_4_p, 'bid_4_s': bid_4_s, 'ask_4_p': ask_4_p, 'ask_4_s': ask_4_s})
dfx.dropna(how='any', axis=0, inplace=True)
dfx['Bids'], dfx['Asks'] = df['B'], df['S']

dfx['bid_volume'] = dfx['bid_0_s'] + dfx['bid_1_s'] + dfx['bid_2_s'] + dfx['bid_3_s'] + dfx['bid_4_s']
dfx['ask_volume'] = dfx['ask_0_s'] + dfx['ask_1_s'] + dfx['ask_2_s'] + dfx['ask_3_s'] + dfx['ask_4_s']

dfx['bid_ask_spread'] = dfx['ask_0_p'] - dfx['bid_0_p']
dfx['time'] = pd.to_datetime(dfx.index)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
style_markdown = {'fontSize': 13, 'margin-top': '1px', 'margin-bottom': '1px', 'margin-left': 'auto',
                  'padding': '0px 5px'}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

graph_cols = ['bid_0_p', 'ask_0_p']
traces = [dict(x=dfx['time'], y=dfx[col], mode='lines', name=dfx[col].name) for col in graph_cols]
traces_spread = [dict(x=dfx['time'], y=dfx['bid_ask_spread'], mode='lines', name='bid-ask-spread')]
traces_bid_ask = [
    dict(x=dfx['bid_0_p'], y=dfx['bid_0_s'], mode='lines+markers', type='scatter', name='bid-price-share'),
    dict(x=dfx['ask_0_p'], y=dfx['ask_0_s'], mode='lines+markers', type='scatter', name='ask-price-share')]
app.layout = html.Div([
    html.Div([
        dcc.Markdown(d(""" ** AKBANK ORDER BOOK 201808** """), style=style_markdown)
    ]),
    html.Div(id='body-div',
             children=[
                 dcc.Graph(id='market-bid-ask',
                           figure={
                               'data': traces,
                               'layout': {
                                   'height': 1000,
                                   'width': 1300,
                                   'margin': {'l': 60, 'b': 30, 'r': 10, 't': 30},
                                   'xaxis': {'showgrid': False, 'title': 'time'},
                                   'yaxis': {'showgrid': False, 'title': 'price'},
                                   'title': 'Market Buy Sell Order'
                               }
                           }),
                 # dcc.Graph(id='spread',
                 #           figure={
                 #               'data': traces_spread,
                 #               'layout': {
                 #                   'height': 350,
                 #                   'width': 3000,
                 #                   'margin': {'l': 20, 'b': 50, 'r': 10, 't': 30},
                 #                   'xaxis': {'showgrid': False},
                 #                   'title': 'Bid Ask Spread'
                 #               }
                 #           }),
                 # dcc.Graph(id='market-level-price-share',
                 #           figure={
                 #               'data': traces_bid_ask,
                 #               'layout': {
                 #                   'title': 'Market Level Price-Share',
                 #                   'height': 350,
                 #                   'width': 1000,
                 #                   'margin': {'l': 20, 'b': 30, 'r': 10, 't': 50},
                 #                   'xaxis': {'showgrid': False},
                 #               }
                 #           },
                 #           style={'display': 'none'})
             ],
             style={'width': '100%', 'display': 'block', 'padding': '0 20',
                    'margin-left': 'auto', 'margin-right': 'auto'})

])

if __name__ == '__main__':
    app.run_server(debug=True)
