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
from plotly.subplots import make_subplots
import seaborn as sns


nav = Navbar()

df = pd.read_csv("dr_auc.csv")

dr_knn=pd.read_csv('dr_knn.csv')



criteria_options = df['criteria'].unique().tolist()
rank_options = df['rank'].unique().tolist()
noise_options =df['noise'].unique().tolist()
sigma_options =df['sigma'].unique().tolist()
plot_summary_options = {'heatmap':'Consistency heatmap across methods',
                        'line':'Consistency across data sets',
                        'bump':'Bump plot of the most consistent methods across data sets',
                        'fit':'Scatter plots of interpretation consistency, predictive consistency, and preditvie accuracy',
                       }
plot_raw_options_knn = {
                   'line_raw':'Consistency vs. predictive accuracy for all data sets',
'k_raw':'Consistency vs. number of local neighbors for all data sets',}


palette = {
            'PCA': 'indigo',
            'Spectral (NN)': 'teal',
            'Spectral (RBF)': 'limegreen',
            'MDS':'slateblue',
#               'NMDS':'skyblue', 
            'Isomap':'magenta',
            't-SNE': 'violet',
            'UMAP':'olivedrab',
            'DAE':'gold',
            'Random Projection':'grey'
            }
meths = list(palette.keys())
line_choice = {
            'PCA': 'solid',
            'Spectral (NN)': 'solid',
            'Spectral (RBF)': 'solid',
            'MDS':'solid',
#             'NMDS':'solid', 
            'Isomap':'dash',
            't-SNE': 'dash',
            'UMAP':'dash',
            'DAE':'dot',
            'Random Projection':'solid'
            }



palette_data = {
    'Statlog':'cornflowerblue',      
    'Spam base':"gold", 
     'WDBC':'slateblue',
     'Tetragonula': 'deepskyblue',
    
    'Author':'salmon',           
    'TCGA':'hotpink',
    'Veronica':"magenta",                   
    'Religion':"firebrick", 
    'PANCAN':"purple",
    'Darmanis':'indigo'
              }

markers_choice = {
                'Random Projection':"0",
                'PCA': "0",
                'MDS':"0",
#                 'NMDS':"0",
                'Spectral (NN)': "0",
                'Spectral (RBF)': "0",
                't-SNE': 'x',
                'UMAP':'x',
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

meths = list(palette.keys())
datas = list(palette_data.keys())


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


def generate_control_card():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            
            

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
                      
            html.Hr(),
            ###############################
            html.P("Select IML questions"),
            dcc.RadioItems(
                id="qq",
                options=[{"label": 'Q1:If we sample a different training set, are the interpretations similar?', "value": 'Q1'}],
                value='Q1',
            ),         
            
            
            ###############################
            html.Hr(),
            html.P("Select: Dimension Rank"),
            dcc.Dropdown(
                id="rank-select_knn",
                options=[{"label": i, "value": i} for i in rank_options],
                
                value=2,
            ),
                
            html.Br(),
            html.Hr(),
            
            html.P("Select: Noise Type"),
            dcc.RadioItems(
                id="noise-select_knn",
                options=[{"label": i, "value": i} for i in noise_options],
                value=noise_options[1],
            ),
                
            html.Hr(),
            html.P("Select: Noise Level (sigma)"),
            dcc.Dropdown(
                id="sigma-select_knn",
                options=[{"label": i, "value": i} for i in sigma_options],
                value=0.15,
            
            ),            
            
            html.Hr(),

            html.P("Select: Consistency Metric"),
            dcc.RadioItems(
                id="criteria-select_knn",
                options=[{"label": i, "value": i} for i in criteria_options],
                value=criteria_options[0],
            ),            

            html.Hr(),

            html.P("Select: Interpretability Method"),
            dcc.Dropdown(
                id="method-select_knn",
                options=[{"label": i, "value": i} for i in meths],
                value=meths[:],
                multi=True,
            ),
            html.Br(),
 

            
            html.Hr(),
            html.P("Select: Data Sets"),
            dcc.Dropdown(
                id="data-select_knn",
                options=[{"label": i, "value": i} for i in datas],
                value=datas[:],
                multi=True,
            ),
            html.Br(),
#             #################################
#             ########### select figures 
#             #################################
#             html.P("Select Summary Graphs you want to show"),
#             dcc.Checklist(id="all_summary",
#                           options=[{"label": 'All', "value":'All_summary' }],value= ['All_summary']),
#             dcc.Checklist(id="select_summary",
#                 options=[{"label": plot_summary_options[i], "value": i} for i in plot_summary_options],
#                 value=list(plot_summary_options.keys()),
#             ),        
            
#             html.Hr(),
#             html.P("Select Raw Graphs you want to show"),
#             dcc.Checklist(id="all_raw",
#                           options=[{"label": 'All', "value":'All_raw' }],value= ['All_raw']),
#             dcc.Checklist(id="select_raw",
#                 options=[{"label": plot_raw_options_knn[i], "value": i} for i in plot_raw_options_knn],
#                 value=list(plot_raw_options_knn.keys()),
#             ),                    


#             html.Hr(),
           
#             dbc.Button('Submit', id='submit-button',n_clicks=100, color="primary",className="me-1"),
#             dbc.Button('Reset',id='reset-button',n_clicks=0, color="secondary",className="me-1"),
            html.Hr(),        ],
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
            html.Div(id='title_summary_knn'),
            html.Div(id='subtitle_summary_knn'),
            ###### summary plots
            html.Div(id='show_heat2_knn'),
            html.Div(id='show_bump_knn'),
            html.Div(id='show_line_knn'),
            ######### raw plots 
            html.Div(id='title_raw_knn'),
            html.Div(id='show_line_raw_knn'),
            html.Div(id='show_k_raw_knn'),
          
           
        ], 
          width={"size": 7, "offset": 1},


            ## raw plots 
#             html.Div(id='show_raw_scatter_clus'),

            
            
        
        )])])
        
    ])
    return layout

def build_line_raw_knn(data_sel, method_sel,criteria_sel,
                  noise_sel,rank_sel,new_data=None
                 ):
   
    aauc=df[
        (df.data.isin(data_sel))
               &(df.method.isin(method_sel))
               &(df.criteria==criteria_sel)
               &(df.noise ==noise_sel)
               &(df['rank'] ==int(rank_sel)

                )
             ]
    this_palette=dict((i,palette[i]) for i in method_sel)
    this_line_choice=dict((i,line_choice[i]) for i in method_sel)
    this_palette_data =  [i for i in palette_data.keys() if i in data_sel]   
    fig = px.line(aauc,x="sigma", y='AUC',color = 'method',

                            color_discrete_map=this_palette,
                                line_dash = 'method',
                      line_dash_map = this_line_choice,
                      labels={
                             "method": "Method"
                         },
                      facet_col="data",facet_col_wrap=3,facet_row_spacing=0.05,
                  #width=1000, height=800,
            category_orders={'data':this_palette_data})
#     fig.update_xaxes(matches=None,showticklabels=True)
    fig.update_xaxes(showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_traces(line=dict(width=3))
    if new_data is not None:
        fig.add_trace(
                go.Scatter(
                    x=neww['data'],
                    y=neww['AUC'],
                    mode='markers',
                    marker=dict(
                        color=[this_palette[i] for i in neww['method']],
                        size=20
                    ),
                    showlegend=False,
                    hoverinfo='none',                                                                              
                )
            )
    fig.update_xaxes(tickangle=45)
    return fig

def build_line_knn(data_sel, method_sel,criteria_sel,
                  noise_sel,sigma_sel,rank_sel
                 ):

####### filter data
    dff=df[
        (df.data.isin(data_sel))
               &(df.method.isin(method_sel))
               &(df.criteria==criteria_sel)
               &(df.noise ==noise_sel)
               &(df.sigma ==float(sigma_sel))
               &(df['rank'] ==int(rank_sel))
             ]
        

    fig = px.line(dff,x="data", y='AUC',color = 'method',markers=True,

                        color_discrete_map=palette,
                            line_dash = 'method',
                  line_dash_map = line_choice,
                  labels={
                         "method": "Method",'data':'Data'
                     },
                 # title=
                 )
    fig.update_xaxes(tickangle=45)
    fig.update_traces(line=dict(width=3))
    fig.update_xaxes(categoryorder='array', categoryarray= datas)
#     fig.add_annotation(dict(font=dict(color="grey",size=12),
#                         x=-0.05, y=-0.1, 
#                         text="Large N",
#                         xref='paper',
#                         yref='paper', 
#                         showarrow=False))
#     fig.add_annotation(dict(font=dict(color="grey",size=12),
#                         x=1.1, y=-0.1, 
#                         text="Large P",
#                         xref='paper',
#                         yref='paper', 
#                         showarrow=False))
    return fig



def build_heat_knn(data_sel, method_sel,criteria_sel,
                  noise_sel,sigma_sel,rank_sel
                 ):

####### filter data
    dff=df[
        (df.data.isin(data_sel))
               &(df.method.isin(method_sel))
               &(df.criteria==criteria_sel)
               &(df.noise ==noise_sel)
               &(df.sigma ==float(sigma_sel))
               &(df['rank'] ==int(rank_sel))
             ]
    this_palette_data=dict((i,palette_data[i]) for i in data_sel)
    sub = dff.pivot("data", "method", "AUC")
    sub=round(sub,3)
    sub=sub[method_sel]
    sub= pd.DataFrame(sub, index=this_palette_data)
    h = px.imshow(sub, text_auto=True, aspect="auto",range_color=(0,1),
                 color_continuous_scale=[(0, "whitesmoke"),(0.33,sns.xkcd_rgb["light teal"]),(0.66, sns.xkcd_rgb["tealish"]),(1, sns.xkcd_rgb["dark cyan"])],
                  origin='lower',labels=dict(x="Method", y="Data", color="AUC"))

    h.update_layout({
    'plot_bgcolor':'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
#     h.layout.height = 500
#     h.layout.width = 1000
    h.update_layout(coloraxis=dict(showscale = False),)
    h.update_xaxes(tickangle=45)

    return h
def build_bump_knn(data_sel, method_sel,
                 criteria_sel,noise_sel,sigma_sel,rank_sel,new_data=None):
####### filter data
    
    dff=df[(df.data.isin(data_sel))
                &(df.method.isin(method_sel))
                &(df.noise ==noise_sel)
                &(df['rank'] ==int(rank_sel))
                &(df.sigma ==float(sigma_sel))
               &(df.criteria==criteria_sel)]
    
    this_palette=dict((i,palette[i]) for i in method_sel)

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
                                                      labels=dict(data="Data",ranking='Rank'),

             )
    fig.update_xaxes(tickangle=45)
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
#     fig.add_annotation(dict(font=dict(color="grey",size=12),
#                         x=-0.05, y=-0.1, 
#                         text="Large N",
#                         xref='paper',
#                         yref='paper', 
#                         showarrow=False))
#     fig.add_annotation(dict(font=dict(color="grey",size=12),
#                         x=1.1, y=-0.1, 
#                         text="Large P",
#                         xref='paper',
#                         yref='paper', 
#                         showarrow=False))
    return fig  
              
                     
def build_k_raw_knn(data_sel, method_sel,criteria_sel,
                 noise_sel,sigma_sel,rank_sel,new_data=None):


    dff=dr_knn[(dr_knn.data.isin(data_sel))
            &(dr_knn.method.isin(method_sel))
                   &(dr_knn.criteria==criteria_sel)
               &(dr_knn.noise ==noise_sel)
            &(dr_knn['rank'] ==rank_sel)
            &(dr_knn.sigma ==float(sigma_sel))]
    this_palette=dict((i,palette[i]) for i in method_sel)
    this_line_choice=dict((i,line_choice[i]) for i in method_sel)
#     this_line_choice= [i for i in line_choice.keys() if i in method_sel]
    this_palette_data =  [i for i in palette_data.keys() if i in data_sel]
    ###### input new data
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        neww = new_data[
            (new_data.noise ==noise_sel)&
             (new_data['rank'] ==rank_sel)&
           
            (new_data.sigma==float(sigma_sel))]
        dff = pd.concat([dff, neww]) 
        for mm in set(new_data['method']):
            this_palette[mm]='black'
            this_line_choice[mm]='solid'
            
    fig = px.line(dff,x="K", y='Consistency',color = 'method',

                            color_discrete_map=this_palette,
                                line_dash = 'method',
                      line_dash_map = this_line_choice,
                      labels={
                        "Consistency": "Jaccard",
                      "method": "Method"
                         },
                      facet_col="data",facet_col_wrap=3,facet_row_spacing=0.15,
                  #width=1000, height=800,
            category_orders={'data':this_palette_data}
                 )
    fig.update_xaxes(matches=None,showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_traces(line=dict(width=3))
#           fig.update_xaxes(tickangle=45)
    
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




