##### import packages 
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from nav import *
#from ipynb.fs.full.navbar import Navbar
nav = Navbar()
body = dbc.Container([
       html.H1('Interpretable Machine Learning'),
       html.H3("Welcome to the Interpretable Machine Learning Dashboard"),

       dbc.Row(
           [
               dbc.Col(
                  [
                    
                     html.P(
                         """\
                        Explore IML reliability. Donec id elit non mi porta gravida at eget metus.Fusce dapibus, tellus ac cursus commodo, tortor mauris condimentumnibh, ut fermentum massa justo sit amet risus. Etiam porta semmalesuada magna mollis euismod. Donec sed odio dui. Donec id elit nonmi porta gravida at eget metus. Fusce dapibus, tellus ac cursuscommodo, tortor mauris condimentum nibh, ut fermentum massa justo sitamet risus. Etiam porta sem malesuada magna mollis euismod. Donec sedodio dui."""

                           ),
                           dbc.Button("Go to Package Github", color="secondary"),
                   ],
                  md=4,
               ),
              dbc.Col(
                 [
                     html.H2("some pretty figure"),
                     dcc.Graph(
                         figure={"data": [{"x": [1, 2, 3], "y": [1, 4, 9]}]}
                            ),
                        ]
                     ),
                 ]
            )
       ],
className="mt-4",
)

def Homepage():
    layout = html.Div([
    nav,
    body
    ])
    return layout

# app = dash.Dash(__name__, external_stylesheets = [dbc.themes.UNITED])
# app.layout = Homepage()
# if __name__ == "__main__":
#     app.run_server()
    