import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from util import display_figure,sync_checklists,parse_contents
from app_ins import App_ins
from app1 import App1,build_scatter,build_bump,build_heat_summary,build_line,build_fit,build_cor,build_line_raw,build_scatter_raw,build_acc_bar,build_heat_raw,build_dot,build_acc_raw,build_heat_consis
from app1_2 import App1_2,build_scatter_reg,build_bump_reg,build_heat_summary_reg,build_line_reg,build_fit_reg,build_cor_reg,build_scatter_raw_reg,build_line_raw_reg,build_acc_bar_reg,build_heat_raw_reg,build_dot_reg,build_acc_raw_reg,build_heat_consis_reg
from app2 import App2,build_scatter_clus,build_bump_clus,build_heat_summary_clus,build_line_clus,build_cor_clus,build_fit_clus,build_line_raw_clus,build_scatter_raw_clus,build_dot_clus,build_acc_bar_clus,build_heat_raw_clus,build_acc_raw_clus,build_heat_consis_clus
from app3 import App3,build_scatter_dr,build_bump_dr,build_heat_summary_dr,build_line_dr,build_cor_dr,build_fit_dr,build_line_raw_dr,build_scatter_raw_dr,build_acc_bar_dr,build_heat_raw_dr,build_dot_dr,build_acc_raw_dr,build_heat_consis_dr
from app3_2 import App3_2,build_line_knn,build_bump_knn,build_line_raw_knn,build_k_raw_knn
from home import Homepage
import plotly.express as px
from dash.dependencies import Input, Output, State, ClientsideFunction
from dash.exceptions import PreventUpdate
meths  = ['LogisticRidge','LogisticLASSO', 'SVM','Tree','RF',
        'XGB', 'deepLIFT (MLP)', 'Integrated Gradients (MLP)',  'Epsilon-LRP (MLP)',
        'Guided Backpropagation (MLP)',           
        'Saliency Maps (MLP)','Occlusion (MLP)',
        'Permutation (LogisticRidge)',
        'Permutation (RF)',  
        'Permutation (XGB)',
          'Permutation (MLP)',       
        
        'Shapley Value (LogisticRidge)' ,
        'Shapley Value (RF)' ,
           'Shapley Value (XGB)' ,
        'Shapley Value (MLP)']

meths2  = ['Ridge','LASSO','SVM',
           'Tree','RF','XGB', 
           'deepLIFT (MLP)', 'Integrated Gradients (MLP)',  'Epsilon-LRP (MLP)',
        'Guided Backpropagation (MLP)',           
        'Saliency Maps (MLP)','Occlusion (MLP)',
        'Permutation (Ridge)',
        'Permutation (RF)',  
        'Permutation (XGB)'  , 
           'Permutation (MLP)',       
        'Shapley Value (Ridge)' ,
        'Shapley Value (RF)'   ,
        'Shapley Value (XGB)',
        'Shapley Value (MLP)' ]

# plot_summary_options = ['heatmap','line','bump','fit','dot','cor']
# plot_summary_new_options = ['line_new','bump_new','fit_new','dot_new','cor_new']
plot_raw_options = ['scatter_raw','line_raw','k_raw','heatmap_raw']
# plot_raw_options_knn = ['line_raw','k_raw']
plot_summary_options = ['heatmap','line','heat2','bump','fit']
plot_summary_new_options = ['line_new','heat2_new','bump_new']
plot_summary_options = {'heatmap':'Consistency heatmap across methods',
                        'line':'Consistency line plot within methods',
                        'heat2':'Consistency heatmap within methods',
                        'bump':'Bump plot of the most consistent methods across data sets',
                        'fit':'Scatter plot of consistency vs. prediction accuracy',

                       }
# dbc.themes.LUX
# dbc.themes.COSMO
# dbc.themes.FLATLY
# dbc.themes.JOURNAL
# dbc.themes.LUMEN
# dbc.themes.MINTY
# dbc.themes.SANDSTONE

# dbc.themes.SOLAR
# dbc.themes.UNITED




app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "IMLreliability"


#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN])
#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED])
app.config.suppress_callback_exceptions = True
app.layout = html.Div([
   html.Div([ html.H3("Can We Trust Machine Learning Interpretations? A Reliability Study"),
    html.H5("Luqin Gan, Genevera Allen")],style={'marginLeft': '10%', 'width': '90%'}),
    
    dcc.Location(id = 'url', refresh = False),
    html.Div(id = 'page-content')
])
server = app.server

@app.callback(Output('page-content', 'children'),
            [Input('url', 'pathname')])

def display_page(pathname):
    if pathname == '/instruction':
        return App_ins()
    if pathname == '/feature_importance_classification':
        return App1()
    if pathname == '/feature_importance_regression':
        return App1_2()
    if pathname == '/clustering':
        return App2()
    if pathname == '/dimension_reduction_clustering':
        return App3()
    if pathname == '/knn':
        return App3_2()
    else:
        return Homepage()
    
@app.callback(
        [
         Output('method-select', 'value')],
        [Input('url', 'pathname'),
            Input('method-select_c', 'value')     
                ],)


def dropdown_options(pathname,radio_value):
    if pathname == '/feature_importance_classification':
        
        if radio_value == 'Selected':
            value = [['LogisticRidge','SVM', 
                    'Permutation (LogisticRidge)',
                         'Shapley Value (LogisticRidge)' ,
                      'Tree','XGB','RF',
                      'Permutation (RF)',  
                       'Shapley Value (RF)' ,
                      'Epsilon-LRP (MLP)',
                    'Guided Backpropagation (MLP)',           
                      'Permutation (MLP)',       
                    'Shapley Value (MLP)']]
        elif radio_value == 'All':
#         options = [{'label': x, 'value': x} for x in method_options]
            value = meths,
        elif radio_value == 'Model Specific':
#         options = [{'label': x, 'value': x} for x in method_options]
            value =  meths[:12],
        else:
#         options = [{'label': x, 'value': x} for x in method_options]
            value =  meths[12:],
    else:
        if radio_value == 'Selected':
               value = [['Ridge','SVM', 
                    'Permutation (Ridge)',
                         'Shapley Value (Ridge)' ,
                      'Tree','XGB','RF',
                      'Permutation (RF)',  
                       'Shapley Value (RF)' ,
                      'Epsilon-LRP (MLP)',
                    'Guided Backpropagation (MLP)',           
                      'Permutation (MLP)',       
                    'Shapley Value (MLP)']]     
        elif radio_value == 'All':
#         options = [{'label': x, 'value': x} for x in method_options]
            value = meths2,
        elif radio_value == 'Model Specific':
#         options = [{'label': x, 'value': x} for x in method_options]
            value =  meths2[:12],
        else:
#         options = [{'label': x, 'value': x} for x in method_options]
            value =  meths2[12:],
    

    return  value

## clustering select split or noise 
@app.callback(Output('controls-container', 'style'), [Input('pert-select_clus', 'value')])
def toggle_container(toggle_value):
    if toggle_value == 'Data Split':
        return {'display': 'none'}
    else:
        return {'display': 'block'}



@app.callback(Output('controls-container_dr', 'style'), [Input('pert-select_dr', 'value')])
def toggle_container(toggle_value):
    if toggle_value == 'Data Split':
        return {'display': 'none'}
    else:
        return {'display': 'block'}

   
    
# ###### select figure 
# @app.callback(
#     Output("select_summary", "value"),
#     Output("all_summary", "value"),
#     Output("select_raw", "value"),
#     Output("all_raw", "value"),
#     Output("reset-button","n_clicks"),
#     Output("submit-button","n_clicks"),

#     Input("select_summary", "value"),
#     Input("all_summary", "value"),
#     Input("select_raw", "value"),
#     Input("all_raw", "value"),
#     Input("reset-button","n_clicks")
# )

# def update_summary_checklists(select_summary, all_summary,select_raw,all_raw,reset):
#     if reset>0:
#         return plot_summary_options,['All_summary' ],plot_raw_options,['All_raw' ],0,1
#     else:
#         new = list(sync_checklists(select_summary, all_summary,plot_summary_options,kind='summary')+sync_checklists(select_raw, all_raw,plot_raw_options,kind='raw'))+[0,1]
# #         new.append(0)
#         return new



    
######## make figures 
# @app.callback(    
#     Output("title_summary", "children"),
#     Output("subtitle_summary", "children"),
#     Output("show_heatmap", "children"),
#     Output("show_line", "children"),
#     Output("show_heat2", "children"),
#     Output("show_bump", "children"),
#     Output("show_fit", "children"),
# #     Output("show_dot", "children"),
# #    Output("show_cor", "children"),

#     [Input('url', 'pathname'),
#     State("select_summary", "value"),
#     Input('submit-button','n_clicks'),        
#     ],
#     prevent_initial_call=False

# )

# def show(pathname,plot_selected,click):
#     if click and click>0 and pathname!='/knn':
#         options = ['heatmap','line','heat2','bump','fit']
# #        options = ['heatmap','line','bump','fit','dot','cor']
#         title = []
#         subtitle = []
#         if len(plot_selected)>0:
#             title=html.H4("Summary Figures", style={"color": "slateblue",'text-align':'center','font-weight': 'bold'})
#             subtitle=html.H5("Aggregated over all data sets", style={"color": "mediumslateblue",'text-align':'center'})
#         return list([title]+[subtitle]+[display_figure(pp,plot_selected,click,pathname) for pp in options])
#     raise PreventUpdate
    
@app.callback(    
    Output("title_summary", "children"),
    Output("subtitle_summary", "children"),
    Output("show_heatmap", "children"),
    Output("show_line", "children"),
    Output("show_heat2", "children"),
    Output("show_bump", "children"),
    Output("show_fit", "children"),

    Output("title_summary_raw", "children"),
    Output("show_line_raw", "children"),
    Output("show_scatter_raw", "children"),
#     Output("show_acc_raw", "children"),
    Output("show_heatmap_raw", "children"),
    [Input('url', 'pathname'),
    Input("qq", "value"),
#     Input('submit-button','n_clicks'),        
    ],
    prevent_initial_call=False

)

  
def show(pathname,qq):

#     if click and click>0 and pathname!='/knn':
    if pathname!='/knn':
        options = ['heatmap','line', 'heat2','bump','fit']
        options2 = ['line_raw','scatter_raw','heatmap_raw']
        if qq=='Q1':
            plot_selected = ['heat2','bump','line']
            plot_selected2=['line_raw']
        if qq=='Q2':
            plot_selected = ['heatmap']  
            plot_selected2=['heatmap_raw']
        if qq=='Q3':
            plot_selected = ['fit']              
            plot_selected2=['scatter_raw']

        title=html.H4("Summary Figures for "+qq, style={"color": "slateblue",'text-align':'center','font-weight': 'bold'})
        subtitle=html.H5("Aggregated over all data sets", style={"color": "mediumslateblue",'text-align':'center'})
        title2=html.H4("Detailed Figures for "+qq, style={"color": "slateblue",'text-align':'center'})
            
        return list([title]+[subtitle]+[display_figure(pp,plot_selected,pathname) for pp in options]+[title2]+
                  [display_figure(pp,plot_selected2,pathname) for pp in options2] )
    raise PreventUpdate
    
    
# @app.callback(    
#     Output("title_summary_raw", "children"),
#     Output("show_line_raw", "children"),
#     Output("show_scatter_raw", "children"),
# #     Output("show_acc_raw", "children"),
#     Output("show_heatmap_raw", "children"),
#     [Input('url', 'pathname'),
#     State("select_raw", "value"),
#      Input('submit-button','n_clicks'),        
#     ],
#     prevent_initial_call=True

# )

# def show_raw(pathname,plot_selected,click):

#     if click and click>0 and pathname!='/knn':
#         options = [,'scatter_raw',']
#         title = []
#         if len(plot_selected)>0:
            
#         return list([title]+[display_figure(pp,plot_selected,click,pathname) for pp in options])
#     raise PreventUpdate    

@app.callback(    
    Output("title_summary_knn", "children"),
    Output("show_line_knn", "children"),
    Output("show_bump_knn", "children"),
    [State('url', 'pathname'),
     State("select_summary", "value"),
     Input('submit-button','n_clicks'),        
    ],
    prevent_initial_call=True)

def show_knn(pathname,plot_selected,click):

    if click and click>0 and pathname=='/knn':
        options = ['line','bump']  
        title = []
        if len(plot_selected)>0:
            title=html.H4("Summary Figures", style={"color": "slateblue",'text-align':'center'})

        return list([title]+[display_figure(pp,plot_selected,click,pathname) for pp in options])
    raise PreventUpdate    
    
@app.callback(    
    Output("title_raw_knn", "children"),
    Output("show_line_raw_knn", "children"),
    Output("show_k_raw_knn", "children"),
    [State('url', 'pathname'),
     State("select_raw", "value"),
     Input('submit-button','n_clicks'),        
    ],
    prevent_initial_call=True)

def show_raw_knn(pathname,plot_selected,click):

    if click and click>0 and pathname=='/knn':
        options = ['line_raw','k_raw']  
        title = []
        if len(plot_selected)>0:
            title=html.H4("Raw Figures", style={"color": "slateblue",'text-align':'center'})

        return list([title]+[display_figure(pp,plot_selected,click,pathname) for pp in options])
    raise PreventUpdate     
########################################
######## make figures for new data 
########################################



@app.callback(Output('output-datatable', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'),
                State('url', 'pathname'))
def update_output(list_of_contents, list_of_names, list_of_dates,pathname):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d,pathname) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

    raise PreventUpdate    

        
@app.callback(    
    Output("new_options", "children"),
    Input('upload-data', 'contents'),
     prevent_initial_call=True

    )

def update_options(contents):
 
    if contents is not None:
        children = html.Div([
            html.P("Select NEW Summary Graphs you want to show"),
            dcc.Checklist(id="all_summary_new",
                          options=[{"label": 'All', "value":'All_summary_new' }],value= []),
            dcc.Checklist(id="select_summary_new",
                          options=[{"label": plot_summary_options[i], "value": i+'_new'} for i in list(plot_summary_options.keys())[1:4]],
                value=[],
            ),        
 
            dbc.Button('Show New Figures', id='submit-button_new',n_clicks = 0, color="primary",className="me-1", size="sm"),
            dbc.Button('Reset New Figures', id='reset-button_new',n_clicks=0, color="secondary",className="me-1", size="sm"),
            html.Br(),
          
            dbc.Button('Remove Data', id='remove-button_new',n_clicks=0, color="warning",className="me-1", size="sm"),
        ])
        return children
    elif len(contents)==0:
        children = html.Div([
            html.P("Data Removed")])
        return children
    
    raise PreventUpdate    

    

@app.callback(    
    Output('upload-data', 'contents'),
    
    State('upload-data', 'contents'),
    Input('remove-button_new','n_clicks'),
    prevent_initial_call=True
    )
  

def remove_new(contents,remove):
    if remove==0:
        return contents
    else:
        return []       
    
@app.callback(
    Output("select_summary_new", "value"),
    Output("all_summary_new", "value"),
    Output("reset-button_new","n_clicks"),
    Output("submit-button_new","n_clicks"),


    Input("select_summary_new", "value"),
    Input("all_summary_new", "value"),
    Input("reset-button_new","n_clicks"),
)

def update_summary_checklists(select_summary, all_summary,reset):
    if reset>0:
        return [],[],0,1
    else:    
        return list(sync_checklists(select_summary, all_summary,plot_summary_new_options,kind='summary_new'))+[0,0]


@app.callback(    
    Output("title_summary_new", "children"),
    Output("show_line_new", "children"),
     Output("show_heat2_new", "children"),
    Output("show_bump_new", "children"),

#     Output("show_fit_new", "children"),

    [ Input('url', 'pathname'),
        Input("select_summary_new", "value"),
        Input('submit-button_new','n_clicks'),        
    ],
    prevent_initial_call=True

)

def show_new(pathname,plot_selected,click):

    if click and click>0 and pathname!='/knn':
        options = ['line_new','heat2_new','bump_new'] 
        title = []
        if len(plot_selected)>0:
            title=html.H4("Summary Figures with New Data", style={"color": "slateblue",'text-align':'center'})

        return list([title]+[display_figure(pp,plot_selected,click,pathname) for pp in options])
        
    raise PreventUpdate    

@app.callback(    
    Output("title_summary_knn_new", "children"),
    Output("show_line_knn_new", "children"),
    Output("show_bump_knn_new", "children"),

    [ State('url', 'pathname'),
        State("select_summary_new", "value"),
        Input('submit-button_new','n_clicks'),        
    ],
    prevent_initial_call=True

)

def show_knn_new(pathname,plot_selected,click):

    if click and click>0 and pathname=='/knn':
        options = ['line_new','heat2_new','bump_new']      
        title = []
        if len(plot_selected)>0:
            title=html.H4("Summary Figures with New Data", style={"color": "slateblue",'text-align':'center'})

        return list([title]+[display_figure(pp,plot_selected,click,pathname) for pp in options])
    
    raise PreventUpdate    
    
    
@app.callback(
    Output("bump", "figure"),
        [Input('url', 'pathname'),
        Input("data-select", "value"),
        Input("method-select", "value"),
        Input("k-select", "value"),
        Input("criteria-select", "value"),

    ],
)

def update_bump(pathname,data_sel, method_sel,
                 k_sel, criteria_sel
 
                 ):
        if pathname == '/feature_importance_classification':
            fig=build_bump(data_sel, method_sel,
                     k_sel, criteria_sel)
            return fig
        if pathname == '/feature_importance_regression':
            fig=build_bump_reg(data_sel, method_sel,
                     k_sel, criteria_sel)
            return fig

    
        

    

@app.callback(
    Output("heatmap", "figure"),
    [Input('url', 'pathname'),
        Input("data-select", "value"),
        Input("method-select", "value"),
        Input("k-select", "value"),
        Input("criteria-select", "value"),
    ],
)

def update_heatmap(pathname,data_sel,method_sel,
                 k_sel, criteria_sel
                #,noise_sel,sigma_sel
                 ):
    
    if pathname == '/feature_importance_classification':
        fig=build_heat_summary(data_sel,method_sel,
                 k_sel, criteria_sel)
        return fig
    if pathname == '/feature_importance_regression':
        fig=build_heat_summary_reg(data_sel,method_sel,
                 k_sel, criteria_sel)
        return fig
    
    
@app.callback(
    Output("heatmap_raw", "figure"),
    [Input('url', 'pathname'),
        Input("data-select", "value"),
        Input("method-select", "value"),
        Input("k-select", "value"),
        Input("criteria-select", "value"),
    ],
)

def update_heatmap_raw(pathname,data_sel,method_sel,
                 k_sel, criteria_sel
                #,noise_sel,sigma_sel
                 ):
    
    if pathname == '/feature_importance_classification':
        fig=build_heat_raw(data_sel, method_sel,
                 k_sel, criteria_sel)
        return fig
    if pathname == '/feature_importance_regression':
        fig=build_heat_raw_reg(data_sel,method_sel,
                 k_sel, criteria_sel)
        return fig
    
# @app.callback(
#     Output("acc_raw", "figure"),
#     [Input('url', 'pathname'),
#         Input("data-select", "value"),
#     ],
# )

# def update_acc_raw(pathname,data_sel):
    
#     if pathname == '/feature_importance_classification':
#         fig=build_acc_raw(data_sel)
#         return fig
#     if pathname == '/feature_importance_regression':
#         fig=build_acc_raw_reg(data_sel)
#         return fig
    
  
    
@app.callback(
    Output("line", "figure"),
    [Input('url', 'pathname'),
        Input("data-select", "value"),
        Input("method-select", "value"),
        Input("k-select", "value"),
        Input("criteria-select", "value"),
    ],
)

def update_line(pathname,data_sel,method_sel,
                 k_sel, criteria_sel
                #,noise_sel,sigma_sel
                 ):
    
    if pathname == '/feature_importance_classification':
        fig=build_line(data_sel, method_sel,
                 k_sel, criteria_sel)
        return fig
    if pathname == '/feature_importance_regression':
        fig=build_line_reg(data_sel,method_sel,
                 k_sel, criteria_sel)
        return fig 
        
@app.callback(
    Output("heat2", "figure"),
    [Input('url', 'pathname'),
        Input("data-select", "value"),
        Input("method-select", "value"),
        Input("k-select", "value"),
        Input("criteria-select", "value"),
    ],
)

def update_heat2(pathname,data_sel,method_sel,
                 k_sel, criteria_sel
                #,noise_sel,sigma_sel
                 ):
    
    if pathname == '/feature_importance_classification':
        fig=build_heat_consis(data_sel, method_sel,
                 k_sel, criteria_sel)
        return fig
    if pathname == '/feature_importance_regression':
        fig=build_heat_consis_reg(data_sel,method_sel,
                 k_sel, criteria_sel)
        return fig 

@app.callback(
    Output("line_raw", "figure"),
    [Input('url', 'pathname'),
        Input("data-select", "value"),
        Input("method-select", "value"),
        Input("k-select", "value"),
        Input("criteria-select", "value"),
    ],
)
    
    
    
def update_line_raw(pathname,data_sel,method_sel,
                 k_sel, criteria_sel
                #,noise_sel,sigma_sel
                 ):
    
    if pathname == '/feature_importance_classification':
        fig=build_line_raw(data_sel, method_sel,
                 k_sel, criteria_sel)
        return fig
    if pathname == '/feature_importance_regression':
        fig=build_line_raw_reg(data_sel,method_sel,
                 k_sel, criteria_sel)
        return fig
        
        

        
@app.callback(
    [Output("fit1", "figure"),
     Output("fit2", "figure"),
       Output("pv1", "figure"),
    Output("pv2", "figure"),
   ],   
    [Input('url', 'pathname'),
        Input("data-select", "value"),
        Input("method-select", "value"),
        Input("k-select", "value"),
        Input("criteria-select", "value"),
    ],
)

def update_fit(pathname,data_sel,method_sel,
                 k_sel, criteria_sel
                 ):
    
    if pathname == '/feature_importance_classification':
        fig1,fig2,pv1,pv2=build_fit(data_sel, method_sel,
                 k_sel, criteria_sel)
        return fig1,fig2,pv1,pv2
    if pathname == '/feature_importance_regression':
        fig1,fig2,pv1,pv2=build_fit_reg(data_sel,method_sel,
                 k_sel, criteria_sel)
        return fig1,fig2,pv1,pv2
    raise PreventUpdate    

# @app.callback(
#     [Output("fit1_new", "figure"),
#      Output("fit2_new", "figure"),
#     ], 
#     [Input('url', 'pathname'),
#         Input("data-select", "value"),
#         Input("method-select", "value"),
#         Input("k-select", "value"),
#         Input("criteria-select", "value"),
#          Input('stored-data', 'data')
#     ],
# )

# def update_fit2(pathname,data_sel, method_sel,
#                  k_sel, criteria_sel,data):

#     if pathname == '/feature_importance_classification':
#         fig1,fig2=build_fit(data_sel, method_sel,
#                  k_sel, criteria_sel,data)
#         return fig1,fig2
#     if pathname == '/feature_importance_regression':
#         fig1,fig2=build_fit_reg(data_sel, method_sel,
#                  k_sel, criteria_sel,data)
#         return fig1,fig2        
#     raise PreventUpdate    
           
#########################

@app.callback(
    Output("scatter_raw", "figure"),
    [Input('url', 'pathname'),
        Input("data-select", "value"),
        Input("method-select", "value"),
        Input("k-select", "value"),
        Input("criteria-select", "value"),
    ],
)

def update_scatter_raw(pathname,data_sel,method_sel,
                 k_sel, criteria_sel
                #,noise_sel,sigma_sel
                 ):
    
    if pathname == '/feature_importance_classification':
        fig=build_scatter_raw(data_sel, method_sel,
                 k_sel, criteria_sel)
        return fig
    if pathname == '/feature_importance_regression':
        fig=build_scatter_raw_reg(data_sel,method_sel,
                 k_sel, criteria_sel)
        return fig 
    raise PreventUpdate    
    

#####################################
@app.callback(
    Output("line_new", "figure"),
        [Input('url', 'pathname'),
        Input("data-select", "value"),
        Input("method-select", "value"),
        Input("k-select", "value"),
        Input("criteria-select", "value"),
         Input('stored-data', 'data')
    ],
)

def update_line2(pathname,data_sel, method_sel,
                 k_sel, criteria_sel,data
                 ):

    if pathname == '/feature_importance_classification':
        fig=build_line(data_sel, method_sel,
                 k_sel, criteria_sel,data)
        return fig
    if pathname == '/feature_importance_regression':
        fig=build_line_reg(data_sel, method_sel,
                 k_sel, criteria_sel,data)
        return fig        
    raise PreventUpdate    
        
@app.callback(
    Output("heat2_new", "figure"),
        [Input('url', 'pathname'),
        Input("data-select", "value"),
        Input("method-select", "value"),
        Input("k-select", "value"),
        Input("criteria-select", "value"),
         Input('stored-data', 'data')
    ],
)

def update_heat22(pathname,data_sel, method_sel,
                 k_sel, criteria_sel,data
                 ):

    if pathname == '/feature_importance_classification':
        fig=build_heat_consis(data_sel, method_sel,
                 k_sel, criteria_sel,data)
        return fig
    if pathname == '/feature_importance_regression':
        fig=build_heat_consis_reg(data_sel, method_sel,
                 k_sel, criteria_sel,data)
        return fig        
    raise PreventUpdate    
    

    
@app.callback(
    Output("bump_new", "figure"),
        [Input('url', 'pathname'),
        Input("data-select", "value"),
        Input("method-select", "value"),
        Input("k-select", "value"),
        Input("criteria-select", "value"),
         Input('stored-data', 'data')
    ],
)

def update_bump2(pathname,data_sel, method_sel,
                 k_sel, criteria_sel,data
                 ):

    if pathname == '/feature_importance_classification':
        fig=build_bump(data_sel, method_sel,
                 k_sel, criteria_sel,data)
        return fig
    if pathname == '/feature_importance_regression':
        fig=build_bump_reg(data_sel, method_sel,
                 k_sel, criteria_sel,data)
        return fig
    
    
# @app.callback(
#     [Output("cor1", "figure"),
#      Output("cor2", "figure"),
#     ],    
# #     Output("cor", "figure"),
#     [Input('url', 'pathname'),
#         Input("data-select", "value"),
#         Input("method-select", "value"),
#         Input("k-select", "value"),
#         Input("criteria-select", "value"),
#     ],
# )

# def update_cor(pathname,data_sel,method_sel,
#                  k_sel, criteria_sel
#                  ):
    
#     if pathname == '/feature_importance_classification':
#         fig1,fig2 =build_cor(data_sel, method_sel,
#                  k_sel, criteria_sel)
#         return fig1,fig2        
#     if pathname == '/feature_importance_regression':
#         fig1,fig2 =build_cor_reg(data_sel,method_sel,
#                  k_sel, criteria_sel)
#         return fig1,fig2        
#     raise PreventUpdate    
    
# @app.callback(
#     [Output("cor1_new", "figure"),
#      Output("cor2_new", "figure"),
#     ],    
# #     Output("cor_new", "figure"),
#         [Input('url', 'pathname'),
#         Input("data-select", "value"),
#         Input("method-select", "value"),
#         Input("k-select", "value"),
#         Input("criteria-select", "value"),
#          Input('stored-data', 'data')
#     ],
# )

# def update_cor2(pathname,data_sel, method_sel,
#                  k_sel, criteria_sel,data
#                  ):

#     if pathname == '/feature_importance_classification':
#         fig1,fig2 =build_cor(data_sel, method_sel,
#                  k_sel, criteria_sel,data)
#         return fig
#     if pathname == '/feature_importance_regression':
#         fig1,fig2 =build_cor_reg(data_sel, method_sel,
#                  k_sel, criteria_sel,data)
#         return fig1,fig2  
#     raise PreventUpdate    
            
    
# @app.callback(
#     [Output("dot1", "figure"),
#      Output("dot2", "figure"),
#     ],
#     [Input('url', 'pathname'),
#         Input("data-select", "value"),
#         Input("method-select", "value"),
#         Input("k-select", "value"),
#         Input("criteria-select", "value"),
#     ],
# )
# def update_dot(pathname,data_sel,method_sel,
#                  k_sel, criteria_sel
#                  ):
#     if pathname == '/feature_importance_classification':    
#         fig1,fig2=build_dot(data_sel, method_sel,
#                     k_sel,
#                     criteria_sel)
#     if pathname == '/feature_importance_regression':    
#         fig1,fig2=build_dot_reg(data_sel, method_sel,
#                     k_sel,
#                     criteria_sel
#                  )
#     return fig1,fig2

# @app.callback(
#     [Output("dot1_new", "figure"),
#      Output("dot2_new", "figure"),
#     ],
#     [Input('url', 'pathname'),
#         Input("data-select", "value"),
#         Input("method-select", "value"),
#         Input("k-select", "value"),
#         Input("criteria-select", "value"),
#          Input('stored-data', 'data')
#     ],
# )
        

# def update_dot2(pathname,data_sel,method_sel,
#                  k_sel, criteria_sel,data
#                  ):
#     if pathname == '/feature_importance_classification':    
    
#         fig1,fig2=build_dot(data_sel, method_sel,
#                     k_sel,
#                     criteria_sel,data
#                  )
#     if pathname == '/feature_importance_regression':    
    
#         fig1,fig2=build_dot_reg(data_sel, method_sel,
#                     k_sel,
#                     criteria_sel,data
#                  )
#     return fig1,fig2    
    
    
######################################
######## clustering 
###########################################
    
@app.callback(
    Output("heatmap_clus", "figure"),
    [
        Input("data-select_clus", "value"),
        Input("method-select_clus", "value"),
        Input("criteria-select_clus", "value"),
        Input("noise-select_clus", "value"),
        Input("sigma-select_clus", "value"),
        Input("pert-select_clus", "value"),
    ],
)

def update_heatmap_clus(data_sel_clus,method_sel_clus,
                    criteria_sel_clus,
                    noise_sel_clus,
                     sigma_sel_clus,pert
                 ):
    if pert != 'Noise Addition':
        fig=build_heat_summary_clus(data_sel_clus,method_sel_clus,
                    criteria_sel_clus,
                    None,
                     None)
    else:
        fig=build_heat_summary_clus(data_sel_clus,method_sel_clus,
                    criteria_sel_clus,
                    noise_sel_clus,
                     sigma_sel_clus)
    return fig
# @app.callback(
#     Output("acc_clus", "figure"),
#     [
#         Input("data-select_clus", "value"),
#         Input("method-select_clus", "value"),
#         Input("criteria-select_clus", "value"),
#         Input("noise-select_clus", "value"),
#         Input("sigma-select_clus", "value"),
#            Input("pert-select_clus", "value"),
#  ],
# )

# def update_acc_bar_clus(data_sel_clus,method_sel_clus,
#                     criteria_sel_clus,
#                     noise_sel_clus,
#                      sigma_sel_clus,pert
#                  ):
#     if pert != 'Noise Addition':
#         fig=build_acc_bar_clus(data_sel_clus,method_sel_clus,
#                     criteria_sel_clus,
#                     None,
#                      None)
#     else:
    
#         fig=build_acc_bar_clus(data_sel_clus,method_sel_clus,
#                     criteria_sel_clus,
#                     noise_sel_clus,
#                      sigma_sel_clus)
#     return fig
    
@app.callback(
    Output("line_clus", "figure"),
    [
        Input("data-select_clus", "value"),
        Input("method-select_clus", "value"),
        Input("criteria-select_clus", "value"),
        Input("noise-select_clus", "value"),
        Input("sigma-select_clus", "value"),
           Input("pert-select_clus", "value"),
 ],
)
   
def update_line_clus(data_sel_clus, method_sel_clus,
                    criteria_sel_clus,
                    noise_sel_clus,
                     sigma_sel_clus,pert
                 ):
    if pert != 'Noise Addition':
        fig=build_line_clus(data_sel_clus, method_sel_clus,
                criteria_sel_clus,
                    None,
                     None)
    else:
    
        fig=build_line_clus(data_sel_clus, method_sel_clus,
                criteria_sel_clus,
                noise_sel_clus,
                 sigma_sel_clus)
    return fig
    
@app.callback(
    Output("heat2_clus", "figure"),
    [
        Input("data-select_clus", "value"),
        Input("method-select_clus", "value"),
        Input("criteria-select_clus", "value"),
        Input("noise-select_clus", "value"),
        Input("sigma-select_clus", "value"),
           Input("pert-select_clus", "value"),
 ],
)
   
def update_heat2_clus(data_sel_clus, method_sel_clus,
                    criteria_sel_clus,
                    noise_sel_clus,
                     sigma_sel_clus,pert
                 ):
    if pert != 'Noise Addition':
        fig=build_heat_consis_clus(data_sel_clus, method_sel_clus,
                criteria_sel_clus,
                    None,
                     None)
    else:
    
        fig=build_heat_consis_clus(data_sel_clus, method_sel_clus,
                criteria_sel_clus,
                noise_sel_clus,
                 sigma_sel_clus)
    return fig
@app.callback(
    Output("line_new_clus", "figure"),
        [
      Input("data-select_clus", "value"),
        Input("method-select_clus", "value"),
        Input("criteria-select_clus", "value"),
        Input("noise-select_clus", "value"),
        Input("sigma-select_clus", "value"),
         Input('stored-data', 'data'),
          Input("pert-select_clus", "value"),
  ],
)

def update_line_clus2(data_sel_clus, 
                        method_sel_clus,
                        criteria_sel_clus,
                        noise_sel_clus,
                        sigma_sel_clus,data,pert
                 ):
    if pert != 'Noise Addition':
        fig=build_line_clus(data_sel_clus, method_sel_clus,
                criteria_sel_clus,
                    None,
                     None,data)
    else:
        fig=build_line_clus(data_sel_clus, 
                        method_sel_clus,
                        criteria_sel_clus,
                        noise_sel_clus,
                        sigma_sel_clus,data)
    return fig  

@app.callback(
    Output("bump_clus", "figure"),
    [
        Input("data-select_clus", "value"),
        Input("method-select_clus", "value"),
        Input("criteria-select_clus", "value"),
        Input("noise-select_clus", "value"),
        Input("sigma-select_clus", "value"),
           Input("pert-select_clus", "value"),
 ],
)
def update_bump_clus(data_sel_clus, 
                     method_sel_clus,
                    criteria_sel_clus,
                    noise_sel_clus,
                     sigma_sel_clus,pert
                 ):
    if pert != 'Noise Addition':
        fig=build_bump_clus(data_sel_clus, method_sel_clus,
                criteria_sel_clus,
                    None,
                     None)
    else:
        fig=build_bump_clus(data_sel_clus, 
                        method_sel_clus,
                        criteria_sel_clus,
                        noise_sel_clus,
                        sigma_sel_clus
                    )
    return fig
    
@app.callback(
    Output("bump_new_clus", "figure"),
        [
      Input("data-select_clus", "value"),
        Input("method-select_clus", "value"),
        Input("criteria-select_clus", "value"),
        Input("noise-select_clus", "value"),
        Input("sigma-select_clus", "value"),
         Input('stored-data', 'data'),
           Input("pert-select_clus", "value"),
 ],
)

def update_bump_clus2(data_sel_clus, 
                        method_sel_clus,
                        criteria_sel_clus,
                        noise_sel_clus,
                        sigma_sel_clus,data,pert
                 ):
    if pert != 'Noise Addition':
        fig=build_bump_clus(data_sel_clus, method_sel_clus,
                criteria_sel_clus,
                    None,
                     None,data)
    else:
        fig=build_bump_clus(data_sel_clus, 
                        method_sel_clus,
                        criteria_sel_clus,
                        noise_sel_clus,
                        sigma_sel_clus,data)
    return fig

    
    
@app.callback(
    [Output("fit1_clus", "figure"),
     Output("fit2_clus", "figure"),
       Output("pv1_clus", "figure"),
    Output("pv2_clus", "figure"),
    ],     
    [Input("data-select_clus", "value"),
        Input("method-select_clus", "value"),
        Input("criteria-select_clus", "value"),
        Input("noise-select_clus", "value"),
        Input("sigma-select_clus", "value"),
          Input("pert-select_clus", "value"),
  ],
)
        

def update_fit_clus(data_sel_clus, method_sel_clus,
                    criteria_sel_clus,
                    noise_sel_clus,
                     sigma_sel_clus,
                    pert
                 ):
    
    if pert != 'Noise Addition':
        fig1,fig2,pv1,pv2=build_fit_clus(data_sel_clus, method_sel_clus,
                criteria_sel_clus,
                    None,
                     None)
    else:
        fig1,fig2,pv1,pv2=build_fit_clus(data_sel_clus, method_sel_clus,
                    criteria_sel_clus,
                    noise_sel_clus,
                     sigma_sel_clus )
    return fig1,fig2,pv1,pv2
@app.callback(
       [Output("fit1_new_clus", "figure"),
     Output("fit2_new_clus", "figure"),
        ], 
        [
      Input("data-select_clus", "value"),
        Input("method-select_clus", "value"),
        Input("criteria-select_clus", "value"),
        Input("noise-select_clus", "value"),
        Input("sigma-select_clus", "value"),
         Input('stored-data', 'data'),
           Input("pert-select_clus", "value"),
 ],
)

def update_fit_clus2(data_sel_clus, 
                        method_sel_clus,
                        criteria_sel_clus,
                        noise_sel_clus,
                        sigma_sel_clus,data,pert
                 ):
    
    if pert != 'Noise Addition':
        fig1,fig2=build_fit_clus(data_sel_clus, method_sel_clus,
                    criteria_sel_clus,
                    None,
                     None,data)
    else:
        fig1,fig2=build_fit_clus(data_sel_clus, 
                        method_sel_clus,
                        criteria_sel_clus,
                        noise_sel_clus,
                        sigma_sel_clus,data)
    return fig1,fig2
    
# @app.callback(
#     [Output("dot1_clus", "figure"),
#      Output("dot2_clus", "figure"),
#     ],
#     [
#         Input("data-select_clus", "value"),
#         Input("method-select_clus", "value"),
#         Input("criteria-select_clus", "value"),
#         Input("noise-select_clus", "value"),
#         Input("sigma-select_clus", "value"),
#           Input("pert-select_clus", "value"),
#   ],
# )
        

# # def update_dot_clus(data_sel_clus, method_sel_clus,
# #                     criteria_sel_clus,
# #                     noise_sel_clus,
# #                      sigma_sel_clus,pert
# #                  ):
    
# #     if pert != 'Noise Addition':
# #         fig1,fig2=build_dot_clus(data_sel_clus, method_sel_clus,
# #                     criteria_sel_clus,
# #                     None,
# #                      None)
# #     else:
# #         fig1,fig2=build_dot_clus(data_sel_clus, method_sel_clus,
# #                     criteria_sel_clus,
# #                     noise_sel_clus,
# #                      sigma_sel_clus
# #                  )
# #     return fig1,fig2
# # @app.callback(
# #     [Output("dot1_new_clus", "figure"),
# #      Output("dot2_new_clus", "figure"),
# #     ],
# #     [
# #         Input("data-select_clus", "value"),
# #         Input("method-select_clus", "value"),
# #         Input("criteria-select_clus", "value"),
# #         Input("noise-select_clus", "value"),
# #         Input("sigma-select_clus", "value"),
# #            Input("pert-select_clus", "value"),
# #  ],
# # )
        

# # def update_dot_clus2(data_sel_clus, method_sel_clus,
# #                     criteria_sel_clus,
# #                     noise_sel_clus,
# #                      sigma_sel_clus,pert
# #                  ):
    
# #     if pert != 'Noise Addition':
# #         fig1,fig2=build_dot_clus(data_sel_clus, method_sel_clus,
# #                     criteria_sel_clus,
# #                     None,
# #                      None)
# #     else:
    
# #         fig1,fig2=build_dot_clus(data_sel_clus, method_sel_clus,
# #                     criteria_sel_clus,
# #                     noise_sel_clus,
# #                      sigma_sel_clus
# #                  )
# #     return fig1,fig2


    
    
# # @app.callback(
# #      [Output("cor1_clus", "figure"),
# #      Output("cor2_clus", "figure"),
# #     ],
# #     #Output("cor_clus", "figure"),
# #     [
# #         Input("data-select_clus", "value"),
# #         Input("method-select_clus", "value"),
# #         Input("criteria-select_clus", "value"),
# #         Input("noise-select_clus", "value"),
# #         Input("sigma-select_clus", "value"),
# #           Input("pert-select_clus", "value"),
# #   ],
# # )

# # def update_cor_clus(data_sel_clus, method_sel_clus,
# #                     criteria_sel_clus,
# #                     noise_sel_clus,
# #                      sigma_sel_clus,pert
# #                  ):
    

# #     if pert != 'Noise Addition':
# #         fig1,fig2 =build_cor_clus(data_sel_clus, method_sel_clus,
# #                     criteria_sel_clus,
# #                     None,
# #                      None)
# #     else:
# #         fig1,fig2 =build_cor_clus(data_sel_clus, method_sel_clus,
# #                     criteria_sel_clus,
# #                     noise_sel_clus,
# #                      sigma_sel_clus)
# #     return fig1,fig2 
# # @app.callback(
# #      [Output("cor1_new_clus", "figure"),
# #      Output("cor2_new_clus", "figure"),
# #     ],
# #     #Output("cor_new_clus", "figure"),
# #         [
# #       Input("data-select_clus", "value"),
# #         Input("method-select_clus", "value"),
# #         Input("criteria-select_clus", "value"),
# #         Input("noise-select_clus", "value"),
# #         Input("sigma-select_clus", "value"),
# #          Input('stored-data', 'data'),
# #           Input("pert-select_clus", "value"),
# #   ],
# # )

# # def update_cor_clus2(data_sel_clus, 
# #                         method_sel_clus,
# #                         criteria_sel_clus,
# #                         noise_sel_clus,
# #                         sigma_sel_clus,data,pert
# #                  ):
# #     if pert != 'Noise Addition':
# #         fig1,fig2 =build_cor_clus(data_sel_clus, method_sel_clus,
# #                     criteria_sel_clus,
# #                     None,
# #                      None,data)
# #     else:
# #         fig1,fig2 =build_cor_clus(data_sel_clus, 
# #                         method_sel_clus,
# #                         criteria_sel_clus,
# #                         noise_sel_clus,
# #                         sigma_sel_clus,data)
# #     return fig1,fig2 
            
    
    
# # @app.callback(
# #     Output("acc_raw_clus", "figure"),
# #     [Input("data-select_clus", "value"),
# #         Input("method-select_clus", "value"),
# #         Input("criteria-select_clus", "value"),
# #         Input("noise-select_clus", "value"),
# #         Input("sigma-select_clus", "value"),
# #            Input("pert-select_clus", "value"),
# #  ],
# # )

# # def update_acc_raw_clus(data_sel, method_sel,
# #                  criteria_sel, noise_sel,sigma_sel,pert
# #                  ):
    
# #     if pert != 'Noise Addition':
# #         fig=build_acc_raw_clus(data_sel,method_sel,
# #                 criteria_sel, None,None)
# #     else:
# #         fig=build_acc_raw_clus(data_sel,method_sel,
# #                 criteria_sel, noise_sel,sigma_sel)
# #     return fig 
# #     raise PreventUpdate       
    
@app.callback(
    Output("scatter_raw_clus", "figure"),
    [Input("data-select_clus", "value"),
        Input("method-select_clus", "value"),
        Input("criteria-select_clus", "value"),
        Input("noise-select_clus", "value"),
        Input("sigma-select_clus", "value"),
           Input("pert-select_clus", "value"),
 ],
)

def update_scatter_raw_clus(data_sel, method_sel,
                 criteria_sel, noise_sel,sigma_sel,pert
                 ):
    
    if pert != 'Noise Addition':
        fig=build_scatter_raw_clus(data_sel,method_sel,
                criteria_sel, None,None)
    else:
        fig=build_scatter_raw_clus(data_sel,method_sel,
                criteria_sel, noise_sel,sigma_sel)
    return fig 
    raise PreventUpdate    
    
@app.callback(
    Output("line_raw_clus", "figure"),
   [Input("data-select_clus", "value"),
        Input("method-select_clus", "value"),
        Input("criteria-select_clus", "value"),
        Input("noise-select_clus", "value"),
        Input("sigma-select_clus", "value"),
#            Input("pert-select_clus", "value"),
 ],
)

def update_line_raw_clus(data_sel, method_sel,
                 criteria_sel, noise_sel,sigma_sel):
    
    fig=build_line_raw_clus(data_sel, method_sel,
                 criteria_sel, noise_sel,sigma_sel)
    return fig
    
    
    
@app.callback(
    Output("heatmap_raw_clus", "figure"),
    [Input("data-select_clus", "value"),
        Input("method-select_clus", "value"),
        Input("criteria-select_clus", "value"),
        Input("noise-select_clus", "value"),
        Input("sigma-select_clus", "value"),
        Input("pert-select_clus", "value")])

def update_heat_raw_clus(data_sel, method_sel,
                 criteria_sel, noise_sel,sigma_sel,pert
                 ):
    
    if pert != 'Noise Addition':
        fig=build_heat_raw_clus(data_sel,method_sel,
                criteria_sel, None,None)
    else:
        fig=build_heat_raw_clus(data_sel,method_sel,
                criteria_sel, noise_sel,sigma_sel)
    return fig             
###############################
        
###### DR page 




        
@app.callback(
    Output("heatmap_dr", "figure"),
    [
        Input("data-select_dr", "value"),
        Input("method-select_dr", "value"),
        Input("criteria-select_dr", "value"),
        Input("noise-select_dr", "value"),
        Input("sigma-select_dr", "value"),
        Input("rank-select_dr", "value"),
          Input("clus-select_dr", "value"),
           Input("pert-select_dr", "value"),
  ],
)
    
def update_heatmap_dr(data_sel_dr,method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                      rank_select_dr,
                      clus_select_dr,
                      pert
                 ):
    
    if pert != 'Noise Addition':
        fig=build_heat_summary_dr(data_sel_dr, method_sel_dr,
                    criteria_sel_dr,
                      None,None,
                      rank_select_dr,
                      clus_select_dr
             )
    else:
        fig=build_heat_summary_dr(data_sel_dr, method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                      rank_select_dr,
                      clus_select_dr)
    return fig    
# @app.callback(
#     Output("acc_dr", "figure"),
#     [
#         Input("data-select_dr", "value"),
#         Input("method-select_dr", "value"),
#         Input("criteria-select_dr", "value"),
#         Input("noise-select_dr", "value"),
#         Input("sigma-select_dr", "value"),
#         Input("rank-select_dr", "value"),
#           Input("clus-select_dr", "value"),
#            Input("pert-select_clus", "value"),
#  ],
# )
    
# def update_acc_bar_dr(data_sel_dr,method_sel_dr,
#                     criteria_sel_dr,
#                     noise_sel_dr,
#                      sigma_sel_dr,
#                       rank_select_dr,
#                       clus_select_dr,
#                                       pert
#                  ):
    
#     if pert != 'Noise Addition':
#         fig=build_acc_bar_dr(data_sel_dr, method_sel_dr,
#                     criteria_sel_dr,
#                       None,None,
#                       rank_select_dr,
#                       clus_select_dr
#              )
#     else:
    
#         fig=build_acc_bar_dr(data_sel_dr, method_sel_dr,
#                     criteria_sel_dr,
#                     noise_sel_dr,
#                      sigma_sel_dr,
#                       rank_select_dr,
#                       clus_select_dr)
#     return fig      
# @app.callback(
#     Output("acc_vs_consis_scatter_dr", "figure"),
#     [
#         Input("data-select_dr", "value"),
#         Input("method-select_dr", "value"),
#         Input("criteria-select_dr", "value"),
#         Input("noise-select_dr", "value"),
#         Input("sigma-select_dr", "value"),
#         Input("rank-select_dr", "value"),
#             Input("clus-select_dr", "value"),
# ],
# )
# def update_scatter_dr(data_sel_dr, 
#                      method_sel_dr,
#                     criteria_sel_dr,
#                     noise_sel_dr,
#                      sigma_sel_dr,
#                       rank_select_dr,
#                       clus_select_dr
#                  ):
#     fig=build_scatter_dr(data_sel_dr, 
#                         method_sel_dr,
#                         criteria_sel_dr,
#                         noise_sel_dr,
#                         sigma_sel_dr,
#                          rank_select_dr,
#                       clus_select_dr
#                     )
#     return fig
    
        
@app.callback(
    Output("bump_dr", "figure"),
    [
        Input("data-select_dr", "value"),
        Input("method-select_dr", "value"),
        Input("criteria-select_dr", "value"),
        Input("noise-select_dr", "value"),
        Input("sigma-select_dr", "value"),
        Input("rank-select_dr", "value"),
            Input("clus-select_dr", "value"),
           Input("pert-select_dr", "value"),
],
)
def update_bump_dr(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                   rank_select_dr,
                      clus_select_dr,
                             pert
                 ):
    
    if pert != 'Noise Addition':
        fig=build_bump_dr(data_sel_dr, method_sel_dr,
                    criteria_sel_dr,
                      None,None,
                      rank_select_dr,
                      clus_select_dr
             )
    else:
        fig=build_bump_dr(data_sel_dr, 
                        method_sel_dr,
                        criteria_sel_dr,
                        noise_sel_dr,
                        sigma_sel_dr,
                      rank_select_dr,
                      clus_select_dr
                    )
    return fig
@app.callback(
    Output("bump_new_dr", "figure"),
        [
     Input("data-select_dr", "value"),
        Input("method-select_dr", "value"),
        Input("criteria-select_dr", "value"),
        Input("noise-select_dr", "value"),
        Input("sigma-select_dr", "value"),
        Input("rank-select_dr", "value"),
           Input("clus-select_dr", "value"),
      Input('stored-data', 'data')
    ],
)

def update_bump_dr2(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                   rank_select_dr,
                      clus_select_dr,data
                 ):
    fig=build_bump_dr(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                   rank_select_dr,
                      clus_select_dr,data)
    return fig


@app.callback(
    Output("line_dr", "figure"),
    [
        Input("data-select_dr", "value"),
        Input("method-select_dr", "value"),
        Input("criteria-select_dr", "value"),
        Input("noise-select_dr", "value"),
        Input("sigma-select_dr", "value"),
        Input("rank-select_dr", "value"),
           Input("clus-select_dr", "value"),
           Input("pert-select_dr", "value"),
 ],
)
def update_line_dr(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                   rank_select_dr,
                      clus_select_dr,pert
                 ):
    if pert != 'Noise Addition':
        fig=build_line_dr(data_sel_dr, method_sel_dr,
                    criteria_sel_dr,
                      None,None,
                      rank_select_dr,
                      clus_select_dr
             )
    else:
        fig=build_line_dr(data_sel_dr, 
                        method_sel_dr,
                        criteria_sel_dr,
                        noise_sel_dr,
                        sigma_sel_dr,
                      rank_select_dr,
                      clus_select_dr
                    )
    return fig


@app.callback(
    Output("heat2_dr", "figure"),
    [
        Input("data-select_dr", "value"),
        Input("method-select_dr", "value"),
        Input("criteria-select_dr", "value"),
        Input("noise-select_dr", "value"),
        Input("sigma-select_dr", "value"),
        Input("rank-select_dr", "value"),
           Input("clus-select_dr", "value"),
           Input("pert-select_dr", "value"),
 ],
)
def update_heat2_dr(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                   rank_select_dr,
                      clus_select_dr,pert
                 ):
    if pert != 'Noise Addition':
        fig=build_heat_consis_dr(data_sel_dr, method_sel_dr,
                    criteria_sel_dr,
                      None,None,
                      rank_select_dr,
                      clus_select_dr
             )
    else:
        fig=build_heat_consis_dr(data_sel_dr, 
                        method_sel_dr,
                        criteria_sel_dr,
                        noise_sel_dr,
                        sigma_sel_dr,
                      rank_select_dr,
                      clus_select_dr
                    )
    return fig

@app.callback(
    Output("line_new_dr", "figure"),
        [
     Input("data-select_dr", "value"),
        Input("method-select_dr", "value"),
        Input("criteria-select_dr", "value"),
        Input("noise-select_dr", "value"),
        Input("sigma-select_dr", "value"),
        Input("rank-select_dr", "value"),
            Input("clus-select_dr", "value"),
     Input('stored-data', 'data')
    ],
)

def update_line_dr2(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                   rank_sel_dr,
                      clus_sel_dr,data
                 ):
    fig=build_line_dr(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                   rank_sel_dr,
                      clus_sel_dr,data)
    return fig



# @app.callback(
#     Output("acc_raw_dr", "figure"),
#     [Input("data-select_dr", "value"),
#         Input("method-select_dr", "value"),
#         Input("criteria-select_dr", "value"),
#         Input("noise-select_dr", "value"),
#         Input("sigma-select_dr", "value"),
#         Input("rank-select_dr", "value"),
#            Input("clus-select_dr", "value"),
#            Input("pert-select_dr", "value"),
#  ],
# )

# def update_acc_raw_dr(data_sel_dr, 
#                      method_sel_dr,
#                     criteria_sel_dr,
#                     noise_sel_dr,
#                      sigma_sel_dr,
#                       rank_sel_dr,
#                       clus_sel_dr,
#                              pert
#                  ):
    
#     if pert != 'Noise Addition':
#         fig =build_acc_raw_dr(data_sel_dr, method_sel_dr,
#                     criteria_sel_dr,
#                       None,None,
#                       rank_sel_dr,
#                       clus_sel_dr
#              )
#     else:
    
#         fig=build_acc_raw_dr(data_sel_dr,method_sel_dr,
#                 criteria_sel_dr, noise_sel_dr,sigma_sel_dr,
#                              rank_sel_dr,
#                       clus_sel_dr)
#     return fig 
#     raise PreventUpdate  
    

@app.callback(
     [Output("fit1_dr", "figure"),
     Output("fit2_dr", "figure"),
       Output("pv1_dr", "figure"),
    Output("pv2_dr", "figure"),    ],   
       [Input("data-select_dr", "value"),
        Input("method-select_dr", "value"),
        Input("criteria-select_dr", "value"),
        Input("noise-select_dr", "value"),
        Input("sigma-select_dr", "value"),
        Input("rank-select_dr", "value"),
           Input("clus-select_dr", "value"),
           Input("pert-select_dr", "value"),
 ],
)

def update_fit_dr(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                      rank_sel_dr,
                      clus_sel_dr,
                             pert
                 ):
    
    if pert != 'Noise Addition':
        fig1,fig2,pv1,pv2 =build_fit_dr(data_sel_dr, method_sel_dr,
                    criteria_sel_dr,
                      None,None,
                      rank_sel_dr,
                      clus_sel_dr
             )
    else:
    

        fig1,fig2,pv1,pv2=build_fit_dr(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                      rank_sel_dr,
                      clus_sel_dr)
    return fig1,fig2,pv1,pv2     
@app.callback(
    [Output("fit1_new_dr", "figure"),
     Output("fit2_new_dr", "figure"),
    ],           [
     Input("data-select_dr", "value"),
        Input("method-select_dr", "value"),
        Input("criteria-select_dr", "value"),
        Input("noise-select_dr", "value"),
        Input("sigma-select_dr", "value"),
        Input("rank-select_dr", "value"),
            Input("clus-select_dr", "value"),
     Input('stored-data', 'data')
    ],
)

def update_fit_dr2(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                   rank_sel_dr,
                      clus_sel_dr,data
                 ):
    fig1,fig2=build_fit_dr(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                   rank_sel_dr,
                      clus_sel_dr,data)
    return fig1,fig2
@app.callback(
    [Output("cor1_dr", "figure"),
     Output("cor2_dr", "figure"),
    ],
    #Output("cor_dr", "figure"),
    [
        Input("data-select_dr", "value"),
        Input("method-select_dr", "value"),
        Input("criteria-select_dr", "value"),
        Input("noise-select_dr", "value"),
        Input("sigma-select_dr", "value"),
        Input("rank-select_dr", "value"),
         Input("clus-select_dr", "value"),
 ],
)

def update_cor_dr(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                      rank_sel_dr,clus_sel_dr
                 ):
    

    fig1,fig2 =build_cor_dr(data_sel_dr, 
                 method_sel_dr,
                criteria_sel_dr,
                noise_sel_dr,
                 sigma_sel_dr,
                  rank_sel_dr,clus_sel_dr)
    return fig1,fig2         
@app.callback(
   [Output("cor1_new_dr", "figure"),
     Output("cor2_new_dr", "figure"),
    ],
    #Output("cor_new_dr", "figure"),
        [
     Input("data-select_dr", "value"),
        Input("method-select_dr", "value"),
        Input("criteria-select_dr", "value"),
        Input("noise-select_dr", "value"),
        Input("sigma-select_dr", "value"),
        Input("rank-select_dr", "value"),
            Input("clus-select_dr", "value"),
        Input('stored-data', 'data')
    ],
)

def update_cor_dr2(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                   rank_sel_dr,clus_sel_dr,data
                 ):
    fig1,fig2 =build_cor_dr(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                   rank_sel_dr,clus_sel_dr,data)
    return fig1,fig2             

@app.callback(
    [Output("dot1_dr", "figure"),
     Output("dot2_dr", "figure"),
    ],
    [
        Input("data-select_dr", "value"),
        Input("method-select_dr", "value"),
        Input("criteria-select_dr", "value"),
        Input("noise-select_dr", "value"),
        Input("sigma-select_dr", "value"),
        Input("rank-select_dr", "value"),
           Input("clus-select_dr", "value"),
 ],
)
        

def update_dot_dr(data_sel_dr, method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,rank_sel_dr,
                      clus_select_dr
                 ):
    
    fig1,fig2=build_dot_dr(data_sel_dr, method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                               rank_sel_dr,
                      clus_select_dr
                 )
    return fig1,fig2
@app.callback(
    [Output("dot1_new_dr", "figure"),
     Output("dot2_new_dr", "figure"),
    ],
    [
        Input("data-select_dr", "value"),
        Input("method-select_dr", "value"),
        Input("criteria-select_dr", "value"),
        Input("noise-select_dr", "value"),
        Input("sigma-select_dr", "value"),
          Input("clus-select_dr", "value"),
      Input('stored-data', 'data')
    ],
)
        

def update_dot_dr2(data_sel_dr, method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,rank_sel_dr,
                      clus_select_dr,data
                 ):
    
    fig1,fig2=build_dot_dr(data_sel_dr, method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,rank_sel_dr,
                      clus_select_dr,data
                 )
    return fig1,fig2


@app.callback(
    Output("scatter_raw_dr", "figure"),
   [
     Input("data-select_dr", "value"),
        Input("method-select_dr", "value"),
        Input("criteria-select_dr", "value"),
        Input("noise-select_dr", "value"),
        Input("sigma-select_dr", "value"),
        Input("rank-select_dr", "value"),
         Input("clus-select_dr", "value"),
              Input("pert-select_dr", "value"),
],
)

def update_scatter_raw_dr(data_sel_dr, method_sel_dr,
                 criteria_sel_dr, noise_sel_dr,sigma_sel_dr,rank_sel_dr,
                      clus_sel_dr,pert):
    
    if pert != 'Noise Addition':
        fig=build_scatter_raw_dr(data_sel_dr, method_sel_dr,
                    criteria_sel_dr,
                      None,None,
                      rank_sel_dr,
                      clus_sel_dr
             )
    else:
    
        fig=build_scatter_raw_dr(data_sel_dr, method_sel_dr,
                 criteria_sel_dr, noise_sel_dr,sigma_sel_dr,rank_sel_dr,
                      clus_sel_dr)
    return fig 
    raise PreventUpdate    
    
@app.callback(
    Output("line_raw_dr", "figure"),
[
     Input("data-select_dr", "value"),
        Input("method-select_dr", "value"),
        Input("criteria-select_dr", "value"),
        Input("noise-select_dr", "value"),
        Input("sigma-select_dr", "value"),
        Input("rank-select_dr", "value"),
          Input("clus-select_dr", "value"),
  ],
)

def update_line_raw_dr(data_sel, method_sel,
                 criteria_sel, noise_sel,sigma_sel,rank_sel,clus_select_dr):
    
    fig=build_line_raw_dr(data_sel, method_sel,
                 criteria_sel, noise_sel,sigma_sel,rank_sel,clus_select_dr)
    return fig

    
@app.callback(
    Output("heatmap_raw_dr", "figure"),
[
     Input("data-select_dr", "value"),
        Input("method-select_dr", "value"),
        Input("criteria-select_dr", "value"),
        Input("noise-select_dr", "value"),
        Input("sigma-select_dr", "value"),
        Input("rank-select_dr", "value"),
        Input("clus-select_dr", "value"),
             Input("pert-select_dr", "value"),
  ],
)

def update_heat_raw_dr(data_sel_dr, method_sel_dr,
                 criteria_sel_dr, noise_sel_dr,sigma_sel_dr,rank_sel_dr,clus_sel_dr,pert):
    
    if pert != 'Noise Addition':
        fig=build_heat_raw_dr(data_sel_dr, method_sel_dr,
                    criteria_sel_dr,
                      None,None,
                      rank_sel_dr,
                      clus_sel_dr
             )
    else:
        fig=build_heat_raw_dr(data_sel_dr, method_sel_dr,
                 criteria_sel_dr, noise_sel_dr,sigma_sel_dr,rank_sel_dr,clus_sel_dr)
    return fig


##########################
########## DR+KNN
#########################

@app.callback(
    Output("line_raw_knn", "figure"),
[
     Input("data-select_knn", "value"),
        Input("method-select_knn", "value"),
         Input("criteria-select_knn", "value"),
       Input("noise-select_knn", "value"),
        Input("rank-select_knn", "value"),
    ],
)

def update_line_raw_knn(data_sel, method_sel,criteria_sel,
                noise_sel,rank_sel):
    
    fig=build_line_raw_knn(data_sel, method_sel,criteria_sel,
                 noise_sel,rank_sel)
    return fig

@app.callback(
    Output("k_raw_knn", "figure"),
[
     Input("data-select_knn", "value"),
        Input("method-select_knn", "value"),
         Input("criteria-select_knn", "value"),
       Input("noise-select_knn", "value"),
        Input("sigma-select_knn", "value"),
        Input("rank-select_knn", "value"),
    ],
)

def update_k_raw_knn(data_sel, method_sel,criteria_sel,
                noise_sel,sigma_sel,rank_sel):
    
    fig=build_k_raw_knn(data_sel, method_sel,criteria_sel,
                 noise_sel,sigma_sel,rank_sel)
    return fig





@app.callback(
    Output("line_knn", "figure"),
    [
        Input("data-select_knn", "value"),
        Input("method-select_knn", "value"),
        Input("criteria-select_knn", "value"),
        Input("noise-select_knn", "value"),
        Input("sigma-select_knn", "value"),
        Input("rank-select_knn", "value"),
    ],
)    

def update_line_knn(data_sel_knn, 
                     method_sel_knn,
                     criteria_sel_knn,
                    noise_sel_knn,
                     sigma_sel_knn,
                   rank_select_knn
                 ):
    fig=build_line_knn(data_sel_knn, 
                        method_sel_knn,
                            criteria_sel_knn,
                 noise_sel_knn,
                        sigma_sel_knn,
                      rank_select_knn
                    )
    return fig
@app.callback(
    Output("line_new_knn", "figure"),
    [
        Input("data-select_knn", "value"),
        Input("method-select_knn", "value"),
        Input("criteria-select_knn", "value"),
        Input("noise-select_knn", "value"),
        Input("sigma-select_knn", "value"),
        Input("rank-select_knn", "value"),
          Input('stored-data', 'data')
   ],
)    

def update_line_knn2(data_sel_knn, 
                     method_sel_knn,
                     criteria_sel_knn,
                       noise_sel_knn,
                     sigma_sel_knn,
                   rank_select_knn,data
                 ):
    fig=build_line_knn(data_sel_knn, 
                        method_sel_knn,
                        criteria_sel_knn,
                    noise_sel_knn,
                        sigma_sel_knn,
                      rank_select_knn,data
                    )
    return fig

@app.callback(
    Output("bump_knn", "figure"),
    [
        Input("data-select_knn", "value"),
        Input("method-select_knn", "value"),
        Input("criteria-select_knn", "value"),
        Input("noise-select_knn", "value"),
        Input("sigma-select_knn", "value"),
        Input("rank-select_knn", "value"),
    ],
)
def update_bump_knn(data_sel_knn, 
                     method_sel_knn,
                    criteria_sel_knn,
                    noise_sel_knn,
                     sigma_sel_knn,
                   rank_select_knn
                 ):
    fig=build_bump_knn(data_sel_knn, 
                        method_sel_knn,
                        criteria_sel_knn,
                        noise_sel_knn,
                        sigma_sel_knn,
                      rank_select_knn
                    )
    return fig
@app.callback(
    Output("bump_new_knn", "figure"),
        [
     Input("data-select_knn", "value"),
        Input("method-select_knn", "value"),
        Input("criteria-select_knn", "value"),
        Input("noise-select_knn", "value"),
        Input("sigma-select_knn", "value"),
        Input("rank-select_knn", "value"),
         Input('stored-data', 'data')
    ],
)

def update_bump_knn2(data_sel_knn, 
                     method_sel_knn,
                    criteria_sel_knn,
                    noise_sel_knn,
                     sigma_sel_knn,
                   rank_select_knn,data
                 ):
    fig=build_bump_dr(data_sel_knn, 
                     method_sel_knn,
                    criteria_sel_knn,
                    noise_sel_knn,
                     sigma_sel_knn,
                   rank_select_knn,data)
    return fig




# @app.callback(Output('output','children'),
#              [Input('submit_button','n_clicks'),
#                  Input('reset_button','n_clicks')])
# def update(input_clicks,submit):
# #     if input_clicks>0:«
#     return f'there are {input_clicks} input clicks. {submit} submits '

# @app.callback(Output('submit_button','n_clicks'),
#              [Input('reset_button','n_clicks')])
#     return 
if __name__ == '__main__':
    app.run_server()