### Data
import pandas as pd
import pickle
### Graphing
import plotly.graph_objects as go
### Dash
import dash
import dash_core_components as dcc
import dash_html_components as html
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

df = pd.read_csv("dr_auc.csv")


data_options = df['data'].unique().tolist()
method_options = df['method'].unique().tolist()
criteria_options = df['criteria'].unique().tolist()
rank_options = df['rank'].unique().tolist()
noise_options =df['noise'].unique().tolist()
sigma_options =df['sigma'].unique().tolist()



palette = {
            'PCA': 'indigo',
            'Spectral (NN)': 'magenta',
            'Spectral (RBF)': 'violet',
            'MDS':'blue',
            #  'NMDS':'cyan', 
            'Isomap':'lime',
            't-SNE': 'green',
            'UMAPP':'limegreen',
            'DAE':'yellow',
            'Random Projection':'grey'
            }
meths = list(palette.keys())
line_choice = {
            'PCA': 'solid',
            'Spectral (NN)': 'solid',
            'Spectral (RBF)': 'solid',
            'MDS':'solid',
            #  'NMDS':'cyan', 
            'Isomap':'dash',
            't-SNE': 'dash',
            'UMAPP':'dash',
            'DAE':'dot',
            'Random Projection':'solid'
            }
palette_data = {
                'PANCAN':"purple",
                'DNase':"firebrick",
                'Religion': 'indigo',
                'Author':'yellow',
                'Spam base':"green",
                'Statlog':"cyan"
                }

markers_choice = {
                'Random Projection':"0",
                'PCA': "0",
                'MDS':"0",
                'NMDS':"0",
                'Spectral (NN)': "0",
                'Spectral (RBF)': "0",
                't-SNE': 'x',
                'UMAPP':'x',
                'Isomap':'x',
                'DAE':'x',

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
plot_summary_options = ['line','bump']
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
                id="data-select_knn",
                options=[{"label": i, "value": i} for i in data_options],
                value=data_options[:],
                multi=True,
            ),
            html.Br(),
            html.Br(),
            html.Hr(),

            html.P("Select Method"),
            dcc.Dropdown(
                id="method-select_knn",
                options=[{"label": i, "value": i} for i in meths],
                value=meths[0:8],
                multi=True,
            ),
            html.Br(),
 
            html.Hr(),
            html.P("Select Rank"),
            dcc.Dropdown(
                id="rank-select_knn",
                options=[{"label": i, "value": i} for i in rank_options],
                
                value=2,
            ),
                
            html.Br(),
            html.Hr(),

            html.P("Select Noise Level (sigma)"),
            dcc.Dropdown(
                id="sigma-select_knn",
                options=[{"label": i, "value": i} for i in sigma_options],
                value=1,
            
            ),
            html.Br(),
            
            
            html.Br(),
            html.Hr(),
            
            html.P("Select Noise Type"),
            dcc.RadioItems(
                id="noise-select_knn",
                options=[{"label": i, "value": i} for i in noise_options],
                value=noise_options[1],
            ),
                
            html.Br(),
            html.Hr(),
            html.P("Select Critetia"),
            dcc.RadioItems(
                id="criteria-select_knn",
                options=[{"label": i, "value": i} for i in criteria_options],
                value=criteria_options[0],
            ),            
            html.Br(),
            html.Br(),
     
        ],
    )            


def App3_2():
    layout = html.Div([
        nav,
        dbc.Container([
   html.H1('Dimension Reduction+KNN'),

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
            html.Div(id='show_line_knn'),
            html.Div(id='show_bump_knn'),
          
           
        ], 
          width={"size": 7, "offset": 1},


            ## raw plots 
#             html.Div(id='show_raw_scatter_clus'),

            
            
        
        )])])
        
    ])
    return layout

def build_raw_knn(data_sel, method_sel,
                 noise_sel,rank_sel
                 ):
   
    aauc=df[
        (df.data.isin(data_sel))
               &(df.method.isin(method_sel))
               &(df.noise ==noise_sel)
               &(df['rank'] ==int(rank_sel))
             ]
        
    
    fig = px.line(aauc, x='sigma', y='AUC', color='method', markers=True,
              facet_col='data',
                 facet_col_wrap=3, 
                color_discrete_map=(palette),
                line_dash='method', line_dash_map= markers_choice,
                 category_orders={"method":list(palette.keys())},
               labels=dict( method="Method")
                )

    fig.update_layout(yaxis_range=[0,1])
    fig.update_xaxes(matches=None)
    fig.for_each_xaxis(lambda xaxis: xaxis.update(showticklabels=True))
    return fig
def build_line_knn(data_sel, method_sel,
                  noise_sel,sigma_sel,rank_sel
                 ):

####### filter data
    dff=df[
        (df.data.isin(data_sel))
               &(df.method.isin(method_sel))
               &(df.noise ==noise_sel)
               &(df.sigma ==float(sigma_sel))
               &(df['rank'] ==int(rank_sel))
             ]
        

    fig = px.line(dff,x="data", y='AUC',color = 'method',markers=True,

                        color_discrete_map=palette,
                            line_dash = 'method',
                  line_dash_map = line_choice,
                  labels={
                         "method": "Method"
                     },
                 # title=
                 )
    fig.update_traces(line=dict(width=3))
    return fig

def build_bump_knn(data_sel, method_sel,
                 criteria_sel,noise_sel,sigma_sel,rank_sel,new_data=None):
####### filter data
    
    dff=df[(df.data.isin(data_sel))
                &(df.method.isin(method_sel))
                &(df.noise ==noise_sel)
                &(df['rank'] ==int(rank_sel))
                &(df.sigma ==float(sigma_sel))
               &(df.criteria==criteria_sel)]
    
    
    this_palette = palette.copy()
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        neww = new_data[(new_data.noise ==noise_sel)
                &(new_data['rank'] ==int(rank_sel))
                &(new_data.sigma ==float(sigma_sel))
               &(new_data.criteria==criteria_sel)]
        dff = pd.concat([dff, neww]) 
        for mm in set(neww['method']):
            this_palette[mm]='black'
##### bump plot 
    df_ave = dff.groupby(['method','noise','sigma','criteria','rank'],as_index=False).mean()
    df_ave['data']='Average'
    df_ave=df_ave[['data','method','AUC']]
    dff=pd.concat([dff,df_ave])

########################
                       
    rankk = dff.sort_values(['AUC'],ascending=False).sort_values(['data','AUC'],ascending=False)[['data','method']]
    rankk['ranking'] = (rankk.groupby('data').cumcount()+1)
    rankk=pd.merge(dff,rankk,how='left',on = ['data','method'])
    top= rankk[rankk["data"] == 'Average'].nsmallest(len(set(rankk['method'])), "ranking")
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
