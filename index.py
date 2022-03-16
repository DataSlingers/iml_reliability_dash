import dash
# import dash_core_components as dcc
# import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from dash import dcc, html

from util import display_figure,sync_checklists,parse_contents
from app_ins import App_ins
from app1 import App1,build_scatter,build_bump,build_heat_summary,build_line,build_fit,build_cor
from app1_2 import App1_2,build_scatter_reg,build_bump_reg,build_heat_summary_reg,build_line_reg,build_fit_reg,build_cor_reg
from app2 import App2,build_scatter_clus,build_bump_clus,build_heat_summary_clus,build_line_clus,build_cor_clus,build_fit_clus
from app3 import App3,build_scatter_dr,build_bump_dr,build_heat_summary_dr,build_line_dr,build_cor_dr,build_fit_dr
from app3_2 import App3_2,build_line_knn,build_raw_knn,build_bump_knn
from home import Homepage
import plotly.express as px
from dash.dependencies import Input, Output, State, ClientsideFunction
from dash.exceptions import PreventUpdate

plot_summary_options = ['heatmap','line','bump','fit','cor']
plot_summary_new_options = ['line_new','bump_new','fit_new','cor_new']
plot_raw_options = ['scatters','lines']

#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED])
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.config.suppress_callback_exceptions = True
app.layout = html.Div([
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
    

###### select figure 
@app.callback(
    Output("select_summary", "value"),
    Output("all_summary", "value"),
    Output("select_raw", "value"),
    Output("all_raw", "value"),
    Output("reset-button","n_clicks"),
    Output("submit-button","n_clicks"),

    Input("select_summary", "value"),
    Input("all_summary", "value"),
    Input("select_raw", "value"),
    Input("all_raw", "value"),
    Input("reset-button","n_clicks")
)

def update_summary_checklists(select_summary, all_summary,select_raw,all_raw,reset):
    if reset>0:
        return [],[],[],[],0,1
    else:
        new = list(sync_checklists(select_summary, all_summary,plot_summary_options,kind='summary')+sync_checklists(select_raw, all_raw,plot_raw_options,kind='raw'))+[0,0]
#         new.append(0)
        return new



    
######## make figures 
@app.callback(    
    Output("show_heatmap", "children"),
    Output("show_line", "children"),
    Output("show_bump", "children"),
    Output("show_fit", "children"),
    Output("show_cor", "children"),

    [Input('url', 'pathname'),
    State("select_summary", "value"),
     Input('submit-button','n_clicks'),        
    ],
    prevent_initial_call=True

)

def show(pathname,plot_selected,click):

    if click and click>0 and pathname!='/knn':
        options = ['heatmap','line','bump','fit','cor']      
        return list([display_figure(pp,plot_selected,click,pathname) for pp in options])
    raise PreventUpdate

@app.callback(    
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
        return [display_figure(pp,plot_selected,click,pathname) for pp in options]
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
                options=[{"label": i, "value": i+'_new'} for i in plot_summary_options[1:]],
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
    Output("show_line_new", "children"),
    Output("show_bump_new", "children"),
    Output("show_fit_new", "children"),
    Output("show_cor_new", "children"),

    [ Input('url', 'pathname'),
        Input("select_summary_new", "value"),
        Input('submit-button_new','n_clicks'),        
    ],
    prevent_initial_call=True

)

def show_new(pathname,plot_selected,click):

    if click and click>0 and pathname!='/knn':
        options = ['line_new','bump_new','fit_new','cor_new']  
        return [display_figure(pp,plot_selected,click,pathname) for pp in options]
    raise PreventUpdate    

@app.callback(    
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
        options = ['line_new','bump_new']      
        return [display_figure(pp,plot_selected,click,pathname) for pp in options]
    
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
    

@app.callback(
    Output("heatmap", "figure"),
    [Input('url', 'pathname'),
        Input("method-select", "value"),
        Input("k-select", "value"),
        Input("criteria-select", "value"),
    ],
)

def update_heatmap(pathname,method_sel,
                 k_sel, criteria_sel
                #,noise_sel,sigma_sel
                 ):
    
    if pathname == '/feature_importance_classification':
        fig=build_heat_summary(method_sel,
                 k_sel, criteria_sel)
        return fig
    if pathname == '/feature_importance_regression':
        fig=build_heat_summary_reg(method_sel,
                 k_sel, criteria_sel)
        return fig
    
    
   
    
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
    Output("fit", "figure"),
    [Input('url', 'pathname'),
        Input("data-select", "value"),
        Input("method-select", "value"),
        Input("k-select", "value"),
        Input("criteria-select", "value"),
    ],
)

def update_fit(pathname,data_sel,method_sel,
                 k_sel, criteria_sel
                #,noise_sel,sigma_sel
                 ):
    
    if pathname == '/feature_importance_classification':
        fig=build_fit(data_sel, method_sel,
                 k_sel, criteria_sel)
        return fig
    if pathname == '/feature_importance_regression':
        fig=build_fit_reg(data_sel,method_sel,
                 k_sel, criteria_sel)
        return fig 
    raise PreventUpdate    

@app.callback(
    Output("fit_new", "figure"),
        [Input('url', 'pathname'),
        Input("data-select", "value"),
        Input("method-select", "value"),
        Input("k-select", "value"),
        Input("criteria-select", "value"),
         Input('stored-data', 'data')
    ],
)

def update_fit2(pathname,data_sel, method_sel,
                 k_sel, criteria_sel,data):

    if pathname == '/feature_importance_classification':
        fig=build_fit(data_sel, method_sel,
                 k_sel, criteria_sel,data)
        return fig
    if pathname == '/feature_importance_regression':
        fig=build_fit_reg(data_sel, method_sel,
                 k_sel, criteria_sel,data)
        return fig        
    raise PreventUpdate    
           
    
    
@app.callback(
    Output("cor", "figure"),
    [Input('url', 'pathname'),
        Input("data-select", "value"),
        Input("method-select", "value"),
        Input("k-select", "value"),
        Input("criteria-select", "value"),
    ],
)

def update_cor(pathname,data_sel,method_sel,
                 k_sel, criteria_sel
                 ):
    
    if pathname == '/feature_importance_classification':
        fig=build_cor(data_sel, method_sel,
                 k_sel, criteria_sel)
        return fig
    if pathname == '/feature_importance_regression':
        fig=build_cor_reg(data_sel,method_sel,
                 k_sel, criteria_sel)
        return fig        
    raise PreventUpdate    
    
@app.callback(
    Output("cor_new", "figure"),
        [Input('url', 'pathname'),
        Input("data-select", "value"),
        Input("method-select", "value"),
        Input("k-select", "value"),
        Input("criteria-select", "value"),
         Input('stored-data', 'data')
    ],
)

def update_cor2(pathname,data_sel, method_sel,
                 k_sel, criteria_sel,data
                 ):

    if pathname == '/feature_importance_classification':
        fig=build_cor(data_sel, method_sel,
                 k_sel, criteria_sel,data)
        return fig
    if pathname == '/feature_importance_regression':
        fig=build_cor_reg(data_sel, method_sel,
                 k_sel, criteria_sel,data)
        return fig        
    raise PreventUpdate    
            

    
    
    
    
######################################
######## clustering 
###########################################
    
@app.callback(
    Output("heatmap_clus", "figure"),
    [
        Input("method-select_clus", "value"),
        Input("criteria-select_clus", "value"),
        Input("noise-select_clus", "value"),
        Input("sigma-select_clus", "value"),
    ],
)

def update_heatmap_clus(method_sel_clus,
                    criteria_sel_clus,
                    noise_sel_clus,
                     sigma_sel_clus
                 ):
    
    fig=build_heat_summary_clus(method_sel_clus,
                    criteria_sel_clus,
                    noise_sel_clus,
                     sigma_sel_clus)
    return fig

    
@app.callback(
    Output("line_clus", "figure"),
    [
        Input("data-select_clus", "value"),
        Input("method-select_clus", "value"),
        Input("criteria-select_clus", "value"),
        Input("noise-select_clus", "value"),
        Input("sigma-select_clus", "value"),
    ],
)
   
def update_line_clus(data_sel_clus, method_sel_clus,
                    criteria_sel_clus,
                    noise_sel_clus,
                     sigma_sel_clus
                 ):

    fig=build_line_clus(data_sel_clus, method_sel_clus,
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
         Input('stored-data', 'data')
    ],
)

def update_line_clus2(data_sel_clus, 
                        method_sel_clus,
                        criteria_sel_clus,
                        noise_sel_clus,
                        sigma_sel_clus,data
                 ):
    fig=build_bump_clus(data_sel_clus, 
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
    ],
)
def update_bump_clus(data_sel_clus, 
                     method_sel_clus,
                    criteria_sel_clus,
                    noise_sel_clus,
                     sigma_sel_clus
                 ):
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
         Input('stored-data', 'data')
    ],
)

def update_bump_clus2(data_sel_clus, 
                        method_sel_clus,
                        criteria_sel_clus,
                        noise_sel_clus,
                        sigma_sel_clus,data
                 ):
    fig=build_bump_clus(data_sel_clus, 
                        method_sel_clus,
                        criteria_sel_clus,
                        noise_sel_clus,
                        sigma_sel_clus,data)
    return fig

    
    
@app.callback(
    Output("fit_clus", "figure"),
    [
        Input("data-select_clus", "value"),
        Input("method-select_clus", "value"),
        Input("criteria-select_clus", "value"),
        Input("noise-select_clus", "value"),
        Input("sigma-select_clus", "value"),
    ],
)
        

def update_fit_clus(data_sel_clus, method_sel_clus,
                    criteria_sel_clus,
                    noise_sel_clus,
                     sigma_sel_clus
                 ):
    
        fig=build_fit_clus(data_sel_clus, method_sel_clus,
                    criteria_sel_clus,
                    noise_sel_clus,
                     sigma_sel_clus
                 )
        return fig


@app.callback(
    Output("fit_new_clus", "figure"),
        [
      Input("data-select_clus", "value"),
        Input("method-select_clus", "value"),
        Input("criteria-select_clus", "value"),
        Input("noise-select_clus", "value"),
        Input("sigma-select_clus", "value"),
         Input('stored-data', 'data')
    ],
)

def update_fit_clus2(data_sel_clus, 
                        method_sel_clus,
                        criteria_sel_clus,
                        noise_sel_clus,
                        sigma_sel_clus,data
                 ):
    fig=build_fit_clus(data_sel_clus, 
                        method_sel_clus,
                        criteria_sel_clus,
                        noise_sel_clus,
                        sigma_sel_clus,data)
    return fig


    
    
@app.callback(
    Output("cor_clus", "figure"),
    [
        Input("data-select_clus", "value"),
        Input("method-select_clus", "value"),
        Input("criteria-select_clus", "value"),
        Input("noise-select_clus", "value"),
        Input("sigma-select_clus", "value"),
    ],
)

def update_cor_clus(data_sel_clus, method_sel_clus,
                    criteria_sel_clus,
                    noise_sel_clus,
                     sigma_sel_clus
                 ):
    

    fig=build_cor_clus(data_sel_clus, method_sel_clus,
                    criteria_sel_clus,
                    noise_sel_clus,
                     sigma_sel_clus)
    return fig        
@app.callback(
    Output("cor_new_clus", "figure"),
        [
      Input("data-select_clus", "value"),
        Input("method-select_clus", "value"),
        Input("criteria-select_clus", "value"),
        Input("noise-select_clus", "value"),
        Input("sigma-select_clus", "value"),
         Input('stored-data', 'data')
    ],
)

def update_cor_clus2(data_sel_clus, 
                        method_sel_clus,
                        criteria_sel_clus,
                        noise_sel_clus,
                        sigma_sel_clus,data
                 ):
    fig=build_cor_clus(data_sel_clus, 
                        method_sel_clus,
                        criteria_sel_clus,
                        noise_sel_clus,
                        sigma_sel_clus,data)
    return fig
            
###############################
        
###### DR page 




        
@app.callback(
    Output("heatmap_dr", "figure"),
    [
        Input("method-select_dr", "value"),
        Input("criteria-select_dr", "value"),
        Input("noise-select_dr", "value"),
        Input("sigma-select_dr", "value"),
        Input("rank-select_dr", "value"),
    ],
)
    
def update_heatmap_dr(method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                      rank_select_dr
                 ):
    
    fig=build_heat_summary_dr(method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                      rank_select_dr)
    return fig    
    
@app.callback(
    Output("acc_vs_consis_scatter_dr", "figure"),
    [
        Input("data-select_dr", "value"),
        Input("method-select_dr", "value"),
        Input("criteria-select_dr", "value"),
        Input("noise-select_dr", "value"),
        Input("sigma-select_dr", "value"),
        Input("rank-select_dr", "value"),
    ],
)
def update_scatter_dr(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                      rank_select_dr
                 ):
    fig=build_scatter_dr(data_sel_dr, 
                        method_sel_dr,
                        criteria_sel_dr,
                        noise_sel_dr,
                        sigma_sel_dr,
                         rank_select_dr
                    )
    return fig
    
        
@app.callback(
    Output("bump_dr", "figure"),
    [
        Input("data-select_dr", "value"),
        Input("method-select_dr", "value"),
        Input("criteria-select_dr", "value"),
        Input("noise-select_dr", "value"),
        Input("sigma-select_dr", "value"),
        Input("rank-select_dr", "value"),
    ],
)
def update_bump_dr(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                   rank_select_dr
                 ):
    fig=build_bump_dr(data_sel_dr, 
                        method_sel_dr,
                        criteria_sel_dr,
                        noise_sel_dr,
                        sigma_sel_dr,
                      rank_select_dr
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
         Input('stored-data', 'data')
    ],
)

def update_bump_dr2(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                   rank_select_dr,data
                 ):
    fig=build_bump_dr(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                   rank_select_dr,data)
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
    ],
)
def update_line_dr(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                   rank_select_dr
                 ):
    fig=build_line_dr(data_sel_dr, 
                        method_sel_dr,
                        criteria_sel_dr,
                        noise_sel_dr,
                        sigma_sel_dr,
                      rank_select_dr
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
         Input('stored-data', 'data')
    ],
)

def update_line_dr2(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                   rank_select_dr,data
                 ):
    fig=build_line_dr(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                   rank_select_dr,data)
    return fig
@app.callback(
    Output("fit_dr", "figure"),
    [
        Input("data-select_dr", "value"),
        Input("method-select_dr", "value"),
        Input("criteria-select_dr", "value"),
        Input("noise-select_dr", "value"),
        Input("sigma-select_dr", "value"),
        Input("rank-select_dr", "value"),
    ],
)

def update_fit_dr(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                      rank_select_dr
                 ):
    

    fig=build_fit_dr(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                      rank_select_dr)
    return fig        
@app.callback(
    Output("fit_new_dr", "figure"),
        [
     Input("data-select_dr", "value"),
        Input("method-select_dr", "value"),
        Input("criteria-select_dr", "value"),
        Input("noise-select_dr", "value"),
        Input("sigma-select_dr", "value"),
        Input("rank-select_dr", "value"),
         Input('stored-data', 'data')
    ],
)

def update_fit_dr2(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                   rank_select_dr,data
                 ):
    fig=build_fit_dr(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                   rank_select_dr,data)
    return fig
@app.callback(
    Output("cor_dr", "figure"),
    [
        Input("data-select_dr", "value"),
        Input("method-select_dr", "value"),
        Input("criteria-select_dr", "value"),
        Input("noise-select_dr", "value"),
        Input("sigma-select_dr", "value"),
        Input("rank-select_dr", "value"),
    ],
)

def update_cor_dr(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                      rank_select_dr
                 ):
    

    fig=build_cor_dr(data_sel_dr, 
                 method_sel_dr,
                criteria_sel_dr,
                noise_sel_dr,
                 sigma_sel_dr,
                  rank_select_dr)
    return fig        
@app.callback(
    Output("cor_new_dr", "figure"),
        [
     Input("data-select_dr", "value"),
        Input("method-select_dr", "value"),
        Input("criteria-select_dr", "value"),
        Input("noise-select_dr", "value"),
        Input("sigma-select_dr", "value"),
        Input("rank-select_dr", "value"),
        Input('stored-data', 'data')
    ],
)

def update_cor_dr2(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                   rank_select_dr,data
                 ):
    fig=build_cor_dr(data_sel_dr, 
                     method_sel_dr,
                    criteria_sel_dr,
                    noise_sel_dr,
                     sigma_sel_dr,
                   rank_select_dr,data)
    return fig            

##########################
########## DR+KNN
#########################

@app.callback(
    Output("knn_raw", "figure"),
    [
        Input("data-select_knn", "value"),
        Input("method-select_knn", "value"),
        Input("noise-select_knn", "value"),
        Input("rank-select_knn", "value"),
    ],
)

def update_knn_raw(data_sel_knn, 
                     method_sel_knn,
                    noise_sel_knn,
                   rank_select_knn
                 ):
    fig=build_raw_knn(data_sel_knn, 
                        method_sel_knn,
                        noise_sel_knn,
                      rank_select_knn
                    )
    return fig


@app.callback(
    Output("line_knn", "figure"),
    [
        Input("data-select_knn", "value"),
        Input("method-select_knn", "value"),
        Input("noise-select_knn", "value"),
        Input("sigma-select_knn", "value"),
        Input("rank-select_knn", "value"),
    ],
)    

def update_line_knn(data_sel_knn, 
                     method_sel_knn,
                    noise_sel_knn,
                     sigma_sel_knn,
                   rank_select_knn
                 ):
    fig=build_line_knn(data_sel_knn, 
                        method_sel_knn,
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
        Input("noise-select_knn", "value"),
        Input("sigma-select_knn", "value"),
        Input("rank-select_knn", "value"),
          Input('stored-data', 'data')
   ],
)    

def update_line_knn2(data_sel_knn, 
                     method_sel_knn,
                    noise_sel_knn,
                     sigma_sel_knn,
                   rank_select_knn,data
                 ):
    fig=build_line_knn(data_sel_knn, 
                        method_sel_knn,
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
# #     if input_clicks>0:
#     return f'there are {input_clicks} input clicks. {submit} submits '

# @app.callback(Output('submit_button','n_clicks'),
#              [Input('reset_button','n_clicks')])
#     return 0

    


if __name__ == '__main__':
    app.run_server()