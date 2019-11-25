from constants import *
from functions import *

stock = 'AKBNK.IS'

df = pd.read_pickle(hist_pkl)[stock]

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children='Whoa, a graph!'),

    html.Div(children='''
        Making a stock graph!.
    '''),

    dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': df.index, 'y': df.Close, 'type': 'line', 'name': stock},
            ],
            'layout': {
                'title': stock
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
