### Data
import pandas as pd
import pickle
### Graphing
import plotly.graph_objects as go
### Dash
import dash
from dash import dcc, html

import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input
## Navbar
from nav import Navbar
import numpy as np
import plotly.express as px
 


nav = Navbar()
# header = html.H3(
#     'Reliability of Feature Importance'
# )

df = pd.read_csv("clustering.csv")
cross = pd.read_csv('cross_clus.csv')
cross_ave=cross.groupby(['method1','method2','criteria','noise','sigma'],as_index=False)['value'].mean()

data_options = df['data'].unique().tolist()
method_options = df['method'].unique().tolist()
criteria_options = df['criteria'].unique().tolist()
noise_options =df['noise'].unique().tolist()
sigma_options =df['sigma'].unique().tolist()


palette = {
            'HC (single)':"firebrick",
            'HC (average)':"magenta",
            'HC (complete)':'indigo',
            'HC (ward)':"violet",
            'K-Means':"blue",
            'K-Means++' :'navy',
            'Gaussian MM':'pink', 
            'Birch':"yellow",
            'Spectral (NN)':"green",
            'Spectral (RBF)':'limegreen',
            'K-Means (minibatch)':"cyan",
    }

line_choice = {
            'HC (single)':"solid",
            'HC (average)':"solid",
            'HC (complete)':'solid',
            'HC (ward)':"solid",
            'K-Means':"dot",
            'K-Means++' :'dot',
            'Gaussian MM':'dot', 
            'Birch':"dot",
            'Spectral (NN)':"dash",
            'Spectral (RBF)':'dash',
            'K-Means (minibatch)':"dot",
    }
meths = list(palette.keys())


palette_data = {'PANCAN':"purple",
                'DNase':"firebrick",
                'Religion': 'indigo',
                'Author':'yellow',
                'Spam base':"green",
                'Statlog':"cyan"
                }

markers_choice = {
                'K-Means':"0",
                'K-Means (minibatch)':"0",
                'K-Means++' :'0',

                'HC (average)':"square",
                'HC (complete)':'square',
                'HC (single)':"square",
                'HC (ward)':"square",
                'Spectral (NN)':"x",
                'Spectral (RBF)':'x',

                'Gaussian MM':'x', 
                'Birch':"x"
                }
def sort(df,column1,sorter1,column2=None,sorter2=None):
    
    df[column1] = df[column1].astype("category")
    df[column1].cat.set_categories(sorter1, inplace=True)
    if column2 is not None:
        df[column2] = df[column2].astype("category")
        df[column2].cat.set_categories(sorter2, inplace=True)
        df=df.sort_values([column1,column2]) 
        df[column2]=df[column2].astype("str")
    else:
        df=df.sort_values([column1]) 
 
    df[column1]=df[column1].astype("str")
    return df

df=sort(df,'data',list(palette_data.keys()),'method',list(palette.keys()))


def description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
#             html.H1("Interpretable Machine Learning"),
            html.Div(
                id="intro",
                children="Explore IML reliability.",
            ),
        ],
    )

plot_summary_options = ['heatmap','line','bump','fit','cor']
plot_raw_options = ['scatters','lines']




def generate_control_card():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            
            #################################
            ########### select figures 
            #################################

            html.Hr(),
            html.P("Select Summary Graphs you want to show"),
            dcc.Checklist(id="all_summary",
                          options=[{"label": 'All', "value":'All_summary' }],value= []),
            dcc.Checklist(id="select_summary",
                options=[{"label": i, "value": i} for i in plot_summary_options],
                value=[],
            ),        
            
            html.Hr(),
            html.P("Select Raw Graphs you want to show"),
            dcc.Checklist(id="all_raw",
                          options=[{"label": 'All', "value":'All_raw' }],value= []),
            dcc.Checklist(id="select_raw",
                options=[{"label": i, "value": i} for i in plot_raw_options],
                value=[],
            ),                    

            

            html.Hr(),
           
            dbc.Button('Submit', id='submit-button',n_clicks=0, color="primary",className="me-1"),
            dbc.Button('Reset',id='reset-button',n_clicks=0, color="secondary",className="me-1"),
            html.Hr(),
   
            #############################
            ####### upload new data
            ############################
            html.P("Upload your reliability results"), 
            
           
            
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            
            html.Div(id='new_options'),
                    
            ###############################
            ###############################
            
            
            
            
            
            
            
            html.Hr(),
            html.P("Select Data"),
            dcc.Dropdown(
                id="data-select_clus",
                options=[{"label": i, "value": i} for i in data_options],
                value=data_options[:6],
                multi=True,
            ),
            html.Br(),
            html.Br(),
            html.Hr(),

            html.P("Select Method"),
            dcc.Dropdown(
                id="method-select_clus",
                options=[{"label": i, "value": i} for i in meths],
                value=meths[0:10],
                multi=True,
            ),
            html.Br(),
            html.Hr(),

   
    
            html.P("Select Noise Level (sigma)"),
            dcc.Dropdown(
                id="sigma-select_clus",
                options=[{"label": i, "value": i} for i in sigma_options],
                value=1,
            
            ),
            html.Br(),
            
            
            html.Br(),
            html.Hr(),
            
            html.P("Select Noise Type"),
            dcc.RadioItems(
                id="noise-select_clus",
                options=[{"label": i, "value": i} for i in noise_options],
                value=noise_options[1],
            ),
                
            html.Br(),
            html.Hr(),
            html.P("Select Critetia"),
            dcc.RadioItems(
                id="criteria-select_clus",
                options=[{"label": i, "value": i} for i in criteria_options],
                value=criteria_options[0],
            ),        
                
            html.Br(),
            html.Br(),
           
        ],
    )     




def App2():
    layout = html.Div([
        nav,
        dbc.Container([
   html.H1('Clustering'),

    dbc.Row([
        dbc.Col([
            # Left column
        html.Div(
            id="left-column",
            children=[description_card(), generate_control_card()],
            className="four columns",
          ),
        ],
            width={"size": 3},

             ),
        dbc.Col(children=[

            html.Div(id='output-datatable'),
            ###### summary plots
            html.Div(id='show_heatmap'),
            html.Div(id='show_line'),
            html.Div(id='show_bump'),
            html.Div(id='show_fit'),
            html.Div(id='show_cor'),
            
           
        ], 
          width={"size": 7, "offset": 1},

            
        )])])
                  
    ])
    return layout


# def sync_checklists(selected, all_selected,options,kind):
#     ctx = callback_context
#     input_id = ctx.triggered[0]["prop_id"].split(".")[0]
#     if "select" in input_id:
#         all_selected = ["All_"+kind] if set(selected) == set(options) else []
            
#     else:
#         selected = options if all_selected else []
#     return selected, all_selected



# def display_clus(pp,plot_selected, click,kind):
#     if click is None:
#         raise PreventUpdate
#     else:
#         heads = {'heatmap':'Interpretations are unreliable (cross methods)',
#                  'line':'Interpretations are unreliable (within methods)',
#                  'bump':'No Free lunch',
#                  'fit':'Accuracy != Consistency',
#                  'cor':'Accuracy != Consistency'
#                 }
#         if pp in plot_selected:
#             return html.Div([
#                     html.B(heads[pp]),
#                     dcc.Graph(id=pp+"_"+kind,
#                               style={'width': '80vh', 'height': '50vh'}),
#                 ])
        
        
        
##################################
########### heatmap

# def display_heatmap_clus(plot,click):
#     if click is None:
#         raise PreventUpdate
#     else:
#         if 'heatmap' in plot:
#             return html.Div([
#                 html.B("Interpretations are unreliable"),
#                 dcc.Graph(id="heatmap_clus",
#                           style={'width': '105vh', 'height': '100vh'}),
#                 ])
        


def build_heat_summary_clus(method_sel,
                 criteria_sel,noise_sel,sigma_sel
                 ):
        sub = cross_ave[(cross_ave['method1'].isin(method_sel))&(cross_ave['method2'].isin(method_sel))]
        sub = sub[(sub['sigma']==sigma_sel)&(sub['criteria']==criteria_sel)&(sub['noise']==noise_sel)]
        sub = sub.pivot("method1", "method2", "value")
        sub = sub.fillna(0)+sub.fillna(0).T
        np.fill_diagonal(sub.values, 1)
        sub=round(sub.reindex(columns=method_sel).reindex(method_sel),3)
        
        
        fig = px.imshow(sub, text_auto=True, aspect="auto",color_continuous_scale='Purp',
               labels=dict(x="Method", y="Method", color="Consistency"))
        return fig
    
    

# ##################################
# ########### lineplot
# #################################
# def display_line_clus(plot,click):
#     if click is None:
#         raise PreventUpdate
#     else:
#         if 'line' in plot:
#             return html.Div([
#                 dcc.Graph(id="line_clus",
#                           style={'width': '105vh', 'height': '100vh'}),
#                 ])
        
def build_line_clus(data_sel, method_sel,
                 criteria_sel, noise_sel,sigma_sel,new_data=None
                 ):

####### filter data
    dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.noise ==noise_sel)
            &(df.sigma ==float(sigma_sel))
            &(df.criteria==criteria_sel)] 
    this_palette = palette.copy()
    this_line_choice= line_choice.copy()
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        neww = new_data[(new_data.noise ==noise_sel)
                &(new_data.sigma ==float(sigma_sel))
               &(new_data.criteria==criteria_sel)]
        dff = pd.concat([dff, neww]) 

    fig = px.line(dff,x="data", y='Consistency',color = 'method',markers=True,

                        color_discrete_map=this_palette,
                            line_dash = 'method',
                  line_dash_map = this_line_choice,
                  labels={
                         "method": "Method"
                     },
                 # title=
                 )
    fig.update_traces(line=dict(width=3))
    if new_data is not None:
        fig.add_trace(
                go.Scatter(
                    x=neww['data'],
                    y=neww['Consistency'],
                    mode='markers',
                    marker=dict(
                        color=[this_palette[i] for i in neww['method']],
                        size=20
                    ),
                    showlegend=False,
                    hoverinfo='none',                                                                              
                )
            )
    return fig

###################

def build_scatter_clus(data_sel, method_sel,
                 criteria_sel,noise_sel,sigma_sel,new_data=None
                 ):
   
    dff=df[(df.data.isin(data_sel))
                &(df.method.isin(method_sel))
                &(df.noise ==noise_sel)
                &(df.sigma ==float(sigma_sel))
                &(df.criteria==criteria_sel)]
    fig = px.scatter(dff, x="Accuracy", y="Consistency", color='method', 
                 facet_col='data',
                 facet_col_wrap=3, 
                color_discrete_map=(palette),
                symbol='method', symbol_map= markers_choice,
                 text = 'method',
                 category_orders={"method":list(palette.keys())},
               labels=dict(Consistency=criteria_sel, method="Method")

                )
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-.2,
        xanchor="right",
        x=1
        ))
    fig.update_traces(  
        
       textposition='top right',
)
    return fig

def build_bump_clus(data_sel, method_sel,
                 criteria_sel,noise_sel,sigma_sel,new_data=None
                 ):
####### filter data
    dff=df[(df.data.isin(data_sel))
                &(df.method.isin(method_sel))
                &(df.noise ==noise_sel)
                &(df.sigma ==float(sigma_sel))
                &(df.criteria==criteria_sel)]
    this_palette = palette.copy()
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        neww = new_data[(new_data.noise ==noise_sel)
                &(new_data.sigma ==float(sigma_sel))
               &(new_data.criteria==criteria_sel)]
        dff = pd.concat([dff, neww]) 
        for mm in set(neww['method']):
            this_palette[mm]='black'        
##### bump plot 
    df_ave = dff.groupby(['method','noise','sigma','criteria'],as_index=False).mean()
    df_ave['data']='Average'
    df_ave=df_ave[['data','method','noise','sigma','criteria','Consistency','Accuracy']]
    dff=pd.concat([dff,df_ave])

########################
                       

    rankk = dff.sort_values(['Consistency'],ascending=False).sort_values(['data','Consistency'],ascending=False)[['data','method']]
    rankk['ranking'] = (rankk.groupby('data').cumcount()+1)
    rankk=pd.merge(dff,rankk,how='left',on = ['data','method'])
    top= rankk[rankk["data"] == 'Average'].nsmallest(len(set(rankk.method)), "ranking")
    rankk['ranking']=[str(i) for i in rankk['ranking']]
    
    fig = px.line(rankk, 
        x="data", y="ranking",
              color='method',markers=True,
             color_discrete_map=this_palette,
              category_orders={"data":list(dff.data.unique()),
                              'ranking':[str(i) for i in range(1,len(set(rankk['ranking']))+1)]
                              },
             )
    fig.update_layout(showlegend=False)
    y_annotation = list(top['method'])[::-1]
    intervals = list(top['ranking'])
    for k in intervals:
        fig.add_annotation(dict(font=dict(color="black",size=10),
                            #x=x_loc,
                            x=1,
                            y=str(k-1),
                            showarrow=False,
                            text=y_annotation[k-1],
                            textangle=0,
                            xanchor='left',
                            xref="paper",
                            yref="y"
                           ))
    fig.update_layout(margin=dict( r=150))
    if new_data is not None:
        new_rankk = rankk[rankk.data.isin(set(neww.data))]
        fig.add_trace(
            go.Scatter(
                x=new_rankk['data'],
                y=new_rankk['ranking'],
                mode='markers',
                marker=dict(
                    color=[this_palette[i] for i in new_rankk['method']],
                    size=20
                ),
                showlegend=False,
                hoverinfo='none',                                                                               )
                    )

    return fig   







def build_fit_clus(data_sel, method_sel,
                 criteria_sel, noise_sel,sigma_sel,new_data=None
                 ):

    dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.noise ==noise_sel)
            &(df.sigma ==float(sigma_sel))
            &(df.criteria==criteria_sel)] 
    this_palette = palette.copy()
    this_markers_choice=markers_choice.copy()
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        neww = new_data[(new_data.noise ==noise_sel)
                &(new_data.sigma ==float(sigma_sel))
               &(new_data.criteria==criteria_sel)]
        dff = pd.concat([dff, neww]) 
        for mm in set(new_data['method']):
            this_palette[mm]='black'
            this_markers_choice[mm]='star'
            
            
    fig = px.scatter(dff, x="Accuracy", y="Consistency", color='method', 
                     trendline="ols",
                color_discrete_map=(this_palette),
                symbol='method', symbol_map= this_markers_choice,
                 category_orders={"method":list(this_palette.keys())},
               labels=dict(Consistency=criteria_sel, method="Method")

                )
   
    fig.update_traces(line=dict(width=3))
    if new_data is not None:
        fig.add_trace(
        go.Scatter(
            x=neww['Accuracy'],
            y=neww['Consistency'],
            mode='markers',
            marker=dict(
                color=[this_palette[i] for i in neww['method']],
                symbol=[this_markers_choice[i] for i in neww['method']], 
                size=20
            ),
            showlegend=False,
            hoverinfo='none'
        )
        )        
    return fig

def build_cor_clus(data_sel, method_sel,
                 criteria_sel, noise_sel,sigma_sel,new_data=None
                 ):

 
    dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.noise ==noise_sel)
            &(df.sigma ==float(sigma_sel))
            &(df.criteria==criteria_sel)] 
    this_palette = palette.copy()
    ###### input new data
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        neww = new_data[(new_data.noise ==noise_sel)
                &(new_data.sigma ==float(sigma_sel))
               &(new_data.criteria==criteria_sel)]
        dff = pd.concat([dff, neww]) 
        for mm in set(new_data['method']):
            this_palette[mm]='black'
            
    corr = dff.groupby(['method'])[['Consistency','Accuracy']].corr().unstack().reset_index()    
    corr.columns = [' '.join(col).strip() for col in corr.columns.values]
    corr=corr[['method','Consistency Accuracy']]
    corr = sort(corr,'method',list(this_palette.keys()))
    
    fig = px.bar(corr, x='method', y='Consistency Accuracy',
             range_y = [-1,1],
             color='method',color_discrete_map=(this_palette),
             labels={'method':'Method', 'Consistency Accuracy':'Correlation'},
             title="Correlation between Accuracy and Consistency"
            )
    fig.show()
    return fig                     
                     




