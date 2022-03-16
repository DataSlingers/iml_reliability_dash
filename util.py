

from dash import Dash, dcc, html, Input, Output, State, callback_context
from dash.exceptions import PreventUpdate
import base64
import datetime
import io
import pandas as pd
from dash import dash_table
import dash_loading_spinners as dls
import dash_bootstrap_components as dbc

def sync_checklists(selected, all_selected,options,kind):
    ctx = callback_context
    input_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if "select" in input_id:
        all_selected = ["All_"+kind] if set(selected) == set(options) else []
            
    else:
        selected = options if all_selected else []
    return selected, all_selected




def display_figure(pp,plot_selected, click,pathname):
    if click is None:
        raise PreventUpdate
 
    if click>0:
        paths = {'/clustering':'clus',
                 '/dimension_reduction_clustering':'dr',
                   '/knn':'knn'
                }
        heads = {'heatmap':'Interpretations are unreliable (cross methods)',
                 'line':'Interpretations are unreliable (within methods)',
                 'bump':'No Free lunch',
                 'fit':'Accuracy != Consistency',
                 'cor':'Accuracy != Consistency',
                 'line_new':'Line with new data',
                 'bump_new':'Bump with new data',
                 'fit_new':'Fit with new data',
                 'cor_new':'Cor with new data',
                 
                }
        if pp in plot_selected:
            fig_id = pp+'_'+paths[pathname] if pathname in paths else pp
           
            return html.Div([
                    html.B(heads[pp]),
                    dls.Hash(                        
                        dcc.Graph(id=fig_id,
                              style={'width': '80vh', 'height': '50vh'}),
                        color="#435278",
                        speed_multiplier=2,
                        size=100,
                    ),
                    html.Hr(),  # horizontal line
                ])
        
########upload data 
def parse_contents(contents, filename, date,pathname):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        elif 'txt' or 'tsv' in filename:
            # Assume that the user upl, delimiter = r'\s+'oaded an excel file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), delimiter = r'\s+')
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    paths = {'/clustering':'clus',
                 '/dimension_reduction':'dr'}
    return html.Div([
        html.H5(filename),
#         html.H6(datetime.datetime.fromtimestamp(date)),
        dash_table.DataTable(
                        data=df.to_dict('records'),
                        columns=[{'name': i, 'id': i} for i in df.columns],
                        page_size=10,
         style_table={'overflowX': 'auto'},
            style_cell={
                'height': 'auto',
                # all three widths are needed
                'minWidth': '50px', 'width': '100px', 'maxWidth': '100px',
                'whiteSpace': 'normal'
            }),
#         dbc.Table.from_dataframe(
#     pd.DataFrame(df.to_dict('records')), striped=True, bordered=True, hover=True, index=False,size='sm',responsive='sm'),

        ##############################
        ### store new data set
        ###############################
        dcc.Store(id='stored-data', data=df.to_dict('records')),
        
          ###### summary plots
        html.Div(id='show_line_new'),
        html.Div(id='show_bump_new'),
        html.Div(id='show_fit_new'),
        html.Div(id='show_cor_new'),

   
#         html.B('Line with new data'),
#         dcc.Graph(id="line_new_"+paths[pathname] if pathname in paths else "line_new",
#                               style={'width': '80vh', 'height': '50vh'}),
#         html.Hr(),  # horizontal line
#         html.B('Bump with new data'),
#         dcc.Graph(id="bump_new_"+paths[pathname] if pathname in paths else "bump_new",
#                               style={'width': '80vh', 'height': '50vh'}),
#         html.Hr(),  # horizontal line
#         html.B('Fit with new data'),
#         dcc.Graph(id="fit_new_"+paths[pathname] if pathname in paths else "fit_new",
#                               style={'width': '80vh', 'height': '50vh'}),
#         html.Hr(),  # horizontal line
#         html.B('Cor with new data'),
#         dcc.Graph(id="cor_new_"+paths[pathname] if pathname in paths else "cor_new",
#                               style={'width': '80vh', 'height': '50vh'}),
#         html.Hr(),  # horizontal line
        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        }),
    ])

        
        