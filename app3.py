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
import dash_loading_spinners as dls
from plotly.subplots import make_subplots
import seaborn as sns



nav = Navbar()
# header = html.H3(
#     'Reliability of Feature Importance'
# )

df = pd.read_csv("dr_clustering.csv")
cross = pd.read_csv('cross_dr.csv')
# accs = pd.read_csv('dr_accs.csv')
df_split = pd.read_csv("dr_clustering_split.csv")
cross_split = pd.read_csv('cross_dr_split.csv')
# accs_split = pd.read_csv('dr_accs_split.csv')


criteria_options = df['criteria'].unique().tolist()
rank_options = df['rank'].unique().tolist()
noise_options =df['noise'].unique().tolist()
sigma_options =df['sigma'].unique().tolist()
clus_options = df['clustering'].unique().tolist()
plot_summary_options = {'heatmap':'Consistency heatmap across methods',
                        'line':'Consistency across data sets',
                        'bump':'Bump plot of the most consistent methods across data sets',
#                        'dot':'Consistency/predictive accuracy vs. methods',
                        'fit':'Scatter plots of interpretation consistency, predictive consistency, and preditvie accuracy',
                       # 'cor': 'Correlation between onsistency and predictive accuracy'
                       }
plot_raw_options = {'scatter_raw':'Consistency vs. number of features for all data sets',
                   'line_raw':'Consistency vs. predictive accuracy for all data sets',
                    'heatmap_raw':'Consistency heatmap across methods for all data sets'}


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
# palette_data = {
#                 'PANCAN':"purple",
#                 'DNase':"firebrick",
#                 'Religion': 'indigo',
#                 'Author':'yellow',
#                 'Spam base':"green",
#                 'Statlog':"cyan"
#                 }
# palette_data = { 
#     'Statlog':'deepskyblue',      
#     'Theorem':'slateblue',   
#     'Spam base':"green", 
#     'Author':'yellow',           
#     'Amphibians':'lightseagreen',  
#     'Madelon' :'greenyellow', 
#     'TCGA':'hotpink',
#     'DNase':"firebrick",                   
#     'Religion': 'indigo',
#    'PANCAN':"purple",
# }



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
df_split=sort(df_split,'data',list(palette_data.keys()),'method',list(palette.keys()))
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
            
            
            html.P("Select IML questions"),
            dcc.RadioItems(
                id="qq",
                options=[{"label": i, "value": j} for (j,i) in [('Q1','Q1:If we sample a different training set, are the interpretations similar?'), ('Q2','Q2: Do two IML methods generate similar interpretations on the same data?'),('Q3','Q3: Does higher accuracy lead to more consistent interpretations?')]],
                value='Q1',
            ),         
            
            html.Hr(),
            
            
            
            html.P("Select: Dimension Rank"),
            dcc.Dropdown(
                id="rank-select_dr",
                options=[{"label": i, "value": i} for i in rank_options],
                
                value=2,
            ),
                ## select data split or noise addition 
            html.P("Select: Pertubation Method"),
            dcc.RadioItems(
                id="pert-select_dr",
                options=[{"label": i, "value": i} for i in ['Data Split','Noise Addition']],
                value='Data Split',
            ),
            
            html.Hr(),            
            
            
            html.Div(id='controls-container_dr', children=[
            
                html.P("Select: Noise Type"),
                dcc.RadioItems(
                    id="noise-select_dr",
                    options=[{"label": i, "value": i} for i in noise_options],
                    value=noise_options[1],
                ),

                html.Hr(),
                html.P("Select: Noise Level (sigma)"),
                dcc.Dropdown(
                    id="sigma-select_dr",
                    options=[{"label": i, "value": i} for i in sigma_options],
                    value=0.5,

                ),
            ]),
                     
                     
            html.Hr(),          

            html.P("Select: Consistency/Accuracy Metric"),
            dcc.RadioItems(
                id="criteria-select_dr",
                options=[{"label": i, "value": i} for i in criteria_options],
                value=criteria_options[0],
            ),  
           

            
            html.Hr(),
            html.P("Select: Clustering Method"),
            dcc.RadioItems(
                id="clus-select_dr",
                options=[{"label": i, "value": i} for i in clus_options],
                value=clus_options[0],
            
            ),
          
            ###############################
            ###############################
            
            html.Hr(),


            html.P("Select: Interpretability Method"),
            dcc.Dropdown(
                id="method-select_dr",
                options=[{"label": i, "value": i} for i in meths],
                value=meths[0:8],
                multi=True,
            ),
            html.Hr(),            
            html.P("Select: Data Sets"),
            dcc.Dropdown(
                id="data-select_dr",
                options=[{"label": i, "value": i} for i in datas],
                value=datas[:],
                multi=True,
            ),
            html.Br(),
            html.Hr(),            
            #################################
            ########### select figures 
            #################################

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
#                 options=[{"label": plot_raw_options[i], "value": i} for i in plot_raw_options],
#                 value=list(plot_raw_options.keys()),
#             ),         
            

#             html.Hr(),
           
#             dbc.Button('Submit', id='submit-button',n_clicks=0, color="primary",className="me-1"),
#             dbc.Button('Reset',id='reset-button',n_clicks=0, color="secondary",className="me-1"),

#             html.Hr(),
        
        
        ],
    )            


def App3():
    layout = html.Div([
        nav,
        dbc.Container([
   html.H1('Dimension Reduction+Clustering'),

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
            html.Div(id='title_summary'),
            html.Div(id='subtitle_summary'),
            html.Div(id='show_heat2'),
            html.Div(id='show_bump'),
            html.Div(id='show_line'),
            html.Div(id='show_heatmap'),
            html.Div(id='show_fit'),
#             html.Div(id='show_dot'),
#            html.Div(id='show_cor'),
            ######### raw plots 
            html.Div(id='title_summary_raw'),
            html.Div(id='show_line_raw'),
            html.Div(id='show_scatter_raw'),
#                html.Div(id='show_acc_raw'),
         html.Div(id='show_heatmap_raw'),
           
        ], 
          width={"size": 7, "offset": 1},
            
        
        )])])
        
    ])
    return layout

def build_scatter_dr(data_sel,method_sel,criteria_sel,noise_sel,sigma_sel,rank_sel,clus_sel):

    this_palette=dict((i,palette[i]) for i in method_sel)
    this_markers_choice =dict((i,markers_choice[i]) for i in method_sel)

    if noise_sel==None:
        dff=df_split[(df_split.data.isin(data_sel))
                    &(df_split['rank'] ==int(rank_sel))
                    &(df_split.clustering == clus_sel)
                     &(df_split.method.isin(method_sel))] 
    else:    
        dff=df[(df.data.isin(data_sel))
                &(df.method.isin(method_sel))
                &(df.noise ==noise_sel)
                &(df.sigma ==float(sigma_sel))
                &(df['rank'] ==int(rank_sel))
               &(df['criteria']==criteria_sel)
            &(df.clustering == clus_sel)
          ]
    fig = px.scatter(dff, x="Accuracy", y="Consistency", color='method', 
                 facet_col='data',
                 facet_col_wrap=3, 
                color_discrete_map=(this_palette),
                symbol='method', symbol_map= this_markers_choice,
                 text = 'method',
                 category_orders={"method":list(this_palette.keys())},
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



def build_heat_consis_dr(data_sel, method_sel,
                 criteria_sel, noise_sel,sigma_sel,rank_sel,clus_sel,new_data=None
                 ):

    if noise_sel==None: ## data spkit
        dff=df_split[(df_split.data.isin(data_sel))
            &(df_split.criteria==criteria_sel)
                      &(df_split.clustering == clus_sel)
                      &(df_split['rank'] ==rank_sel)
                    &(df_split.method.isin(method_sel))] 
    else:
            
        dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.noise ==noise_sel)
            &(df.sigma ==float(sigma_sel))
            &(df['rank'] ==rank_sel)
            &(df.clustering == clus_sel)
            &(df.criteria==criteria_sel)] 

    this_palette=dict((i,palette[i]) for i in method_sel)
    this_line_choice=dict((i,line_choice[i]) for i in method_sel)
    this_palette_data=dict((i,palette_data[i]) for i in data_sel)
    ###### input new data
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        if noise_sel==None:
            neww = new_data[(new_data.criteria==criteria_sel)
                              &(new_data['rank'] ==int(rank_sel))]
            
        else:
            new_data['sigma']=[float(i) for i in new_data['sigma']]
            neww = new_data[(new_data.noise ==noise_sel)
                              &(new_data['rank'] ==int(rank_sel))
                      &(new_data.sigma ==float(sigma_sel))
                   &(new_data.criteria==criteria_sel)]


        dff = pd.concat([dff, neww],join='inner', ignore_index=True) 
        for mm in set(new_data['method']):
            this_palette[mm]='black'
            this_line_choice[mm]='solid'
            method_sel=method_sel+[mm]
        for mm in set(new_data['data']):
            this_palette_data[mm]='black'    
            data_sel=data_sel+[mm]   
    sub = dff.pivot("data", "method", "Consistency")
    sub=round(sub,3)
    sub= pd.DataFrame(sub, index=this_palette_data)
    sub=sub[method_sel]
    h = px.imshow(sub, text_auto=True, aspect="auto",range_color=(0,1),
                           color_continuous_scale=[(0, "whitesmoke"),(0.33,sns.xkcd_rgb["light teal"]),(0.66, sns.xkcd_rgb["tealish"]),(1, sns.xkcd_rgb["dark cyan"])],
                  origin='lower',labels=dict(x="Method", y="Data", color="Consistency"))

    h.update_layout({
    'plot_bgcolor':'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    h.layout.height = 500
    h.layout.width = 1000
    
    
    dff_ac = dff[["data", "method", "Accuracy"]].drop_duplicates()
    sub = dff_ac.pivot("data", "method", "Accuracy")
    sub=round(sub,3)
    sub=sub[method_sel]
    sub= pd.DataFrame(sub, index=this_palette_data)
    h2=px.imshow(sub, text_auto=True, aspect="auto", color_continuous_scale=[(0, "whitesmoke"),(0.33,sns.xkcd_rgb["light teal"]),(0.66, sns.xkcd_rgb["tealish"]),(1, sns.xkcd_rgb["dark cyan"])],
                 range_color=(0,1),
                 origin='lower',labels=dict(x="Method", y="Data", color="Consistency"))
    h2.layout.height = 500
    h2.layout.width = 700

    fig= make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], 
                                horizontal_spacing=0.15,
                            vertical_spacing=0.05,
                       subplot_titles=('Interpretation Consistency','Clustering Accuracy'))

    for trace in h.data:
        fig.add_trace(trace, 1, 1)
    for trace in h2.data:
        fig.add_trace(trace, 1, 2)
    fig.update_xaxes(tickangle=45)
    # for trace in bar1.data:
    fig.update_layout(
                  coloraxis=dict(colorscale =[(0, "whitesmoke"),(0.33,sns.xkcd_rgb["light teal"]),(0.66,sns.xkcd_rgb["tealish"]),(1, sns.xkcd_rgb["dark cyan"])],showscale = False))
    fig['layout']['yaxis2']['title']='Data'
    fig['layout']['xaxis2']['title']='Method'    
    fig['layout']['yaxis']['title']='Data'
    fig['layout']['xaxis']['title']='Method'
    return fig
def build_bump_dr(data_sel, method_sel,
                 criteria_sel,noise_sel,sigma_sel,rank_sel,clus_sel,new_data=None):
####### filter data
    if noise_sel==None:
        dff=df_split[(df_split.data.isin(data_sel))
             &(df_split['rank'] ==int(rank_sel))
                                 &(df_split.criteria==criteria_sel)
   &(df_split.clustering == clus_sel)
                     &(df_split.method.isin(method_sel))] 
     
    else:
        dff=df[(df.data.isin(data_sel))
                &(df.method.isin(method_sel))
                &(df.noise ==noise_sel)
                &(df['rank'] ==int(rank_sel))
                &(df.sigma ==float(sigma_sel))
               &(df.criteria==criteria_sel)
                      &(df.clustering == clus_sel)
                ]
        
    
        
    df_ave = dff.groupby(['method'],as_index=False).mean()
    df_ave['data']='Average'
    df_ave=df_ave[['data','method','Consistency','Accuracy']]
    dff=pd.concat([dff,df_ave])
    this_palette=dict((i,palette[i]) for i in method_sel)
    this_markers_choice=dict((i,markers_choice[i]) for i in method_sel)
    this_palette_data=dict((i,palette_data[i]) for i in data_sel)

    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        if noise_sel==None:
            neww = new_data[(new_data.criteria==criteria_sel)
                              &(new_data['rank'] ==int(rank_sel))]
            
        else:
            new_data['sigma']=[float(i) for i in new_data['sigma']]
            neww = new_data[(new_data.noise ==noise_sel)
                              &(new_data['rank'] ==int(rank_sel))
                      &(new_data.sigma ==float(sigma_sel))
                   &(new_data.criteria==criteria_sel)]


        dff = pd.concat([dff, neww],join='inner', ignore_index=True) 
        for mm in set(new_data['method']):
            this_palette[mm]='black'
            method_sel=method_sel+[mm]
        for mm in set(new_data['data']):
            this_palette_data[mm]='black'    
            data_sel=data_sel+[mm]   
########################
                       
    rankk = dff.sort_values(['Consistency'],ascending=False).sort_values(['data','Consistency'],ascending=False)[['data','method']]
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
        new_rankk = rankk[(rankk.data.isin(set(neww.data)))&(rankk.method.isin(set(neww.method)))]
        
        fig.add_trace(
            go.Scatter(
                x=new_rankk['data'],
                y=new_rankk['ranking'],
                mode='markers',
                marker=dict(
                    color='black',
                    size=15,
                    line=dict(
                            color='MediumPurple',
                            width=5
                                ),
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




    
    
    
#     if noise_sel!=None:


#             cross_ave = cross[cross.data.isin(data_sel) &(cross['clustering'] == clus_sel)&(cross['rank']==rank_sel)
#                              &(cross['sigma']==sigma_sel)&(cross['criteria']==criteria_sel)&(cross['noise']==noise_sel)]
#         else:
#             cross_ave = cross_split[cross_split.data.isin(data_sel) &(cross_split['clustering'] == clus_sel)
#                                     &(cross_split['criteria']==criteria_sel)&(cross_split['rank']==rank_sel)]
            
#         sub = cross_ave[(cross_ave['method1'].isin(method_sel))&(cross_ave['method2'].isin(method_sel))]
#         sub=sub.groupby(['method1','method2'],as_index=False)['value'].mean()
#         sub = sub.pivot("method1", "method2", "value")
#         sub = sub.fillna(0)+sub.fillna(0).T
#         np.fill_diagonal(sub.values, 1)
#         sub=round(sub.reindex(columns=method_sel).reindex(method_sel),3)
               
#         fig = px.imshow(sub, text_auto=True, origin='lower',
#                         aspect="auto",color_continuous_scale='
#                         labels=dict(x="Method", y="Method", color="Consistency"))        
#         fig.layout.coloraxis.showscale = False
#         return fig

def build_acc_bar_dr(data_sel, method_sel,criteria_sel,noise_sel,sigma_sel,rank_sel,clus_sel
                 ):
    this_palette=dict((i,palette[i]) for i in method_sel)
    dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.noise ==noise_sel)
            &(df.sigma ==float(sigma_sel))
            &(df['rank'] ==rank_sel)
            &(df.criteria==criteria_sel)
                      &(df.clustering == clus_sel)
                ] 

    dff = dff.groupby(['method']).mean().reset_index()
    fig = px.bar(dff, x='method', y='Accuracy',
                 range_y = [0,1],
                 color='method',text_auto='.3',
                 color_discrete_map=this_palette,
                 labels={'method':'Method', 'Accuracy':'Predictive Accuracy'},
                 title="Predictive Accuracy"
                )
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(height=300,showlegend=False)

    return fig

def build_line_dr(data_sel, method_sel,
                 criteria_sel, noise_sel,sigma_sel,rank_sel,clus_sel,new_data=None
                 ):

    if noise_sel==None: ## data spkit
        dff=df_split[(df_split.data.isin(data_sel))
            &(df_split.criteria==criteria_sel)
                      &(df_split.clustering == clus_sel)
                      &(df_split['rank'] ==rank_sel)
                    &(df_split.method.isin(method_sel))] 
    else:
            
        dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.noise ==noise_sel)
            &(df.sigma ==float(sigma_sel))
            &(df['rank'] ==rank_sel)
            &(df.clustering == clus_sel)
            &(df.criteria==criteria_sel)] 

    this_palette=dict((i,palette[i]) for i in method_sel)
    this_line_choice=dict((i,line_choice[i]) for i in method_sel)
    this_palette_data=dict((i,palette_data[i]) for i in data_sel)
    ###### input new data
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        new_data=new_data.replace({'normal':'Normal','laplace':'Laplace'})
        if noise_sel==None:
            neww = new_data[(new_data.criteria==criteria_sel)&(new_data['rank'] ==rank_sel)]
            
        else:
            new_data['sigma']=[float(i) for i in new_data['sigma']]
            neww = new_data[(new_data.noise ==noise_sel)
                    &(new_data.sigma ==float(sigma_sel))
                            &(new_data['rank'] ==rank_sel)
                   &(new_data.criteria==criteria_sel)]
        dff = pd.concat([dff, neww],join='inner', ignore_index=True) 

        for mm in set(new_data['method']):
            this_palette[mm]='black'
            this_line_choice[mm]='dash'
            method_sel=method_sel+[mm]
        for mm in set(new_data['data']):
            this_palette_data[mm] = 'black'
            data_sel=data_sel+[mm]
    fig1 = px.line(dff,x="data", y='Consistency',color = 'method',markers=True,

                        color_discrete_map=this_palette,
#                             line_dash = 'method',
#                   line_dash_map = this_line_choice,
                  labels={
                             "method": "Method",'data':'Data'
                     },
                 # title=
                 )
    if new_data is not None:
        fig1.add_trace(
                go.Scatter(
                    x=neww['data'],
                    y=neww['Consistency'],
                    mode='markers',
                    marker=dict(
                        color='black',
                        size=15,
                        line=dict(
                                color='MediumPurple',
                                width=5
                                    ),
                ),
                    showlegend=False,
                    hoverinfo='none',                                                                              
                )
            )
    
    fig2 = px.line(dff,x="data", y='Accuracy',color = 'method',markers=True,

#                             color_discrete_map=this_palette,
#                       line_dash_map = this_line_choice,
                      labels={
                             "method": "Method",'data':'Data'
                         },
                     )


    if new_data is not None:
        fig2.add_trace(
                go.Scatter(
                    x=neww['data'],
                    y=neww['Accuracy'],
                    mode='markers',
                    marker=dict(
                        color='black',
                        size=15,
                        line=dict(
                                color='MediumPurple',
                                width=5
                                    ),
                ),
                    showlegend=False,
                    hoverinfo='none',                                                                              
                )
            )
    
    
    for i in range(len(fig2['data'])):
        fig2['data'][i]['showlegend']=False
        
    fig= make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], 
                                horizontal_spacing=0.15,
                            vertical_spacing=0.05,   subplot_titles=('Interpretation Consistency','Prediction Accuracy'))
    for trace in fig1.data:
        fig.add_trace(trace, 1, 1)
    for trace in fig2.data:
        fig.add_trace(trace, 1, 2)
    fig.update_traces(line=dict(width=3))
    
    fig['layout']['xaxis2']['title']='Data'
    fig['layout']['yaxis2']['title']='Consistency'    
    fig['layout']['xaxis']['title']='Data'
    fig['layout']['yaxis']['title']='Consistency'
#     fig.add_annotation(dict(font=dict(color="grey",size=12),
#                         x=-0.05, y=-0.1, 
#                         text="Large N",
#                         xref='paper',
#                         yref='paper', 
#                         showarrow=False))
#     fig.add_annotation(dict(font=dict(color="grey",size=12),
#                         x=0.55, y=-0.1, 
#                         text="Large P",
#                         xref='paper',
#                         yref='paper', 
#                         showarrow=False))
    fig.update_xaxes(categoryorder='array', categoryarray= datas)

    fig.update_xaxes(tickangle=45)
     
    return fig

def build_acc_raw_clus(data_sel, method_sel,criteria_sel,noise_sel,sigma_sel,rank_sel,clus_sel,new_data=None):
    if noise_sel==None:
        acc=accs_split[(accs_split.data.isin(data_sel))
            &(accs_split.method.isin(method_sel))
            &(accs_split['rank'] ==rank_sel)
            &(accs_split.criteria==criteria_sel)
               &(accs_split.clustering == clus_sel)
                    ] 
    else:
        acc=accs[(accs.data.isin(data_sel))
                  &(accs['method'].isin(method_sel))
                 &(accs.noise==noise_sel)
                 &(accs_split['rank'] ==rank_sel)
                &(accs_split.criteria==criteria_sel)
               &(accs_split.clustering == clus_sel)
                 &(accs.sigma==sigma_sel)]

    fig = px.box(acc, x="method", y="Accuracy", color='method', 
                     facet_col="data",facet_col_wrap=3,facet_row_spacing=0.1,facet_col_spacing=0.05,
                        color_discrete_map =palette, 
               labels=dict(data='Data',Accuracy='Accuracy', method="Method"))
   
    fig.update_traces(marker_size=10)
    fig.update_xaxes(matches=None,showticklabels=True)
    
    return fig
def build_fit_dr(data_sel, method_sel,
                 criteria_sel, noise_sel,sigma_sel,rank_sel,clus_sel,new_data=None
                 ):
    if noise_sel==None:
        dff=df_split[(df_split.data.isin(data_sel))
            &(df_split.method.isin(method_sel))
            &(df_split['rank'] ==rank_sel)
            &(df_split.criteria==criteria_sel)
               &(df.clustering == clus_sel)
                    ] 
    else:
            
        dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.noise ==noise_sel)
            &(df.sigma ==float(sigma_sel))
            &(df['rank'] ==rank_sel)
            &(df.criteria==criteria_sel)
               &(df.clustering == clus_sel)] 

    
    
    this_palette=dict((i,palette[i]) for i in method_sel)
    this_markers_choice=dict((i,markers_choice[i]) for i in method_sel)
    this_palette_data=dict((i,palette_data[i]) for i in data_sel)

    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        neww = new_data[(new_data.noise ==noise_sel)
                &(new_data['rank'] ==int(rank_sel))
                &(new_data.sigma ==float(sigma_sel))
               &(new_data.criteria==criteria_sel)]
        dff = pd.concat([dff, neww
                        ]) 
        for mm in set(new_data['method']):
            this_palette[mm]='black'
            this_markers_choice[mm]='star'
        
        
        
            
    fig1 = px.scatter(dff, x="Consistency", y="Accuracy", color='data', 
                     trendline="ols",
                color_discrete_map=this_palette_data,
                 category_orders={"Data":list(this_palette_data.keys())},
               labels=dict(Consistency='Consistency', data="Data"),
                custom_data=['data','method'],
                )            
    fig1.update_traces(
        hovertemplate="<br>".join([
        "Data: %{customdata[0]}",
        "Method: %{customdata[1]}",
        "Accuracy: %{y}",
        "Consistency: %{x}",
            ]))   
    fig1.update_traces(line=dict(width=3),marker = dict(size=10),opacity=0.9)
    
    region_lst = []
    pv1 = pd.DataFrame(columns = ['data','p-values'])

    for trace in fig1["data"]:
        trace["name"] = trace["name"].split(",")[0]

        if trace["name"] not in region_lst and trace["marker"]['symbol'] == 'circle':
            trace["showlegend"] = True
            region_lst.append(trace["name"])
        else:
            trace["showlegend"] = False
    model = px.get_trendline_results(fig1)
    nn=len(fig1['data'])//2
    for i in range(0,len(fig1['data']),2):
        fig1["data"][i+1]['customdata'] = fig1["data"][i]['customdata']
        order = np.argsort(fig1["data"][i]['x'])
        fig1["data"][i]['x'] = fig1["data"][i]['x'][order]
        fig1["data"][i]['y'] = fig1["data"][i]['y'][order]


        
        results = model.iloc[i//2]["px_fit_results"]
        alpha = results.params[0]
        beta = results.params[1]
        p_beta = results.pvalues[1]
        r_squared = results.rsquared

        line1 = 'y = ' + str(round(alpha, 4)) + ' + ' + str(round(beta, 4))+'x'
        line2 = 'p-value = ' + '{:.5f}'.format(p_beta)
        line3 = 'R^2 = ' + str(round(r_squared, 3))
        # summary = line1 + '<br>' + line2 + '<br>' + line3
        fitted = np.repeat([[line1,line2,line3]], len(fig1["data"][i+1]['x']), axis=0)
        fig1["data"][i+1]['customdata']=np.column_stack((fig1["data"][i+1]['customdata'],fitted))
        fig1["data"][i+1]['hovertemplate'] = 'Data: %{customdata[0]}<br>Method: %{customdata[1]}<br> %{customdata[2]} <br> %{customdata[3]}<br> %{customdata[4]}'
        if beta<0 and p_beta*nn<0.05:
            pv1.loc[len(pv1)]=[fig1["data"][i+1]['name'],str(min(round(p_beta*nn,3),1))+ ' (Negative)']
        else:
            pv1.loc[len(pv1)]=[fig1["data"][i+1]['name'],min(round(p_beta*nn,3),1)]


            
    if new_data is not None:
        fig1.add_trace(
        go.Scatter(
            y=neww['Accuracy'],
            x=neww['Consistency'],
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
        
    fig2 = px.scatter(dff, x="Consistency", y="Accuracy", color='method', 
                     trendline="ols",
                color_discrete_map=this_palette_data,
                 category_orders={"Method":list(this_palette.keys())},
               labels=dict(Consistency='Consistency', method="Method"),
                custom_data=['data','method'],
                )            
        
    fig2.update_traces(
        hovertemplate="<br>".join([
        "Data: %{customdata[0]}",
        "Method: %{customdata[1]}",
        "Accuracy: %{y}",
        "Consistency: %{x}",
            ]))   
    fig2.update_traces(line=dict(width=3),marker = dict(size=10),opacity=0.9)
    
    region_lst2 = []
    pv2 = pd.DataFrame(columns = ['data','p-values'])

    for trace in fig2["data"]:
        trace["name"] = trace["name"].split(",")[0]

        if trace["name"] not in region_lst2 and trace["marker"]['symbol'] == 'circle':
            trace["showlegend"] = True
            region_lst2.append(trace["name"])
        else:
            trace["showlegend"] = False
    model2 = px.get_trendline_results(fig2)
    nn=len(fig2['data'])//2
    for i in range(0,len(fig2['data']),2):
        fig2["data"][i+1]['customdata'] = fig2["data"][i]['customdata']
        order = np.argsort(fig2["data"][i]['x'])
        fig2["data"][i]['x'] = fig2["data"][i]['x'][order]
        fig2["data"][i]['y'] = fig2["data"][i]['y'][order]


        
        results2 = model2.iloc[i//2]["px_fit_results"]
        alpha = results2.params[0]
        beta = results2.params[1]
        p_beta = results2.pvalues[1]
        r_squared = results2.rsquared

        line1 = 'y = ' + str(round(alpha, 4)) + ' + ' + str(round(beta, 4))+'x'
        line2 = 'p-value = ' + '{:.5f}'.format(p_beta)
        line3 = 'R^2 = ' + str(round(r_squared, 3))
        # summary = line1 + '<br>' + line2 + '<br>' + line3
        fitted = np.repeat([[line1,line2,line3]], len(fig2["data"][i+1]['x']), axis=0)
        fig2["data"][i+1]['customdata']=np.column_stack((fig2["data"][i+1]['customdata'],fitted))
        fig2["data"][i+1]['hovertemplate'] = 'Data: %{customdata[0]}<br>Method: %{customdata[1]}<br> %{customdata[2]} <br> %{customdata[3]}<br> %{customdata[4]}'
        if beta<0 and p_beta*nn<0.05:
            pv2.loc[len(pv2)]=[fig2["data"][i+1]['name'],str(min(round(p_beta*nn,3),1))+ ' (Negative)']
        else:
            pv2.loc[len(pv2)]=[fig2["data"][i+1]['name'],min(round(p_beta*nn,3),1)]


    if new_data is not None:
        fig2.add_trace(
        go.Scatter(
            y=neww['Accuracy'],
            x=neww['Consistency'],
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
    pv1=round(pv1,3)
    fig_pv1 = go.Figure(data=[go.Table(header=dict(values=['Data','Interpretation Consistency vs. Prediction Accuracy',
            ]),
                 cells=dict(values=np.array(pv1.T)))
                     ])
    fig_pv1.update_layout(title_text='P-values of fitted line (bonferroni corrected)')
    pv2=round(pv2,3)
    fig_pv2 = go.Figure(data=[go.Table(header=dict(values=['Data','Interpretation Consistency vs. Prediction Accuracy',
            ]),
                 cells=dict(values=np.array(pv2.T)))
                     ])
    fig_pv2.update_layout(title_text='P-values of fitted line (bonferroni corrected)')
    
    return fig1,fig2,fig_pv1,fig_pv2
def build_acc_raw_dr(data_sel, method_sel,
                 criteria_sel, noise_sel,sigma_sel,rank_sel,clus_sel,new_data=None
                 ):

    if noise_sel==None:
        acc=accs_split[(accs_split.data.isin(data_sel)) 
                       &(accs_split['method'].isin(method_sel))
                       &(accs_split['clustering']==(clus_sel))
                       &(accs_split['rank']==(rank_sel))
                        &(accs_split.criteria==criteria_sel)]
    else:
        acc=accs[(accs.data.isin(data_sel))
                  &(accs['method'].isin(method_sel))
                   &(accs['clustering']==(clus_sel))
                  &(accs['rank']==(rank_sel))  
                 &(accs.noise==noise_sel)
                 &(accs.criteria==criteria_sel)
                 &(accs.sigma==sigma_sel)]

    fig = px.box(acc, x="method", y="Accuracy", color='method', 
                     facet_col="data",facet_col_wrap=3,facet_row_spacing=0.1,
                        color_discrete_map =palette, 
               labels=dict(data='Data',Accuracy='Accuracy', method="Method"))
   
    fig.update_traces(marker_size=10)
    fig.update_xaxes(matches=None,showticklabels=True)
    
    return fig 

def build_cor_dr(data_sel, method_sel,
                 criteria_sel, noise_sel,sigma_sel,rank_sel,clus_sel,new_data=None
                 ):

    if noise_sel==None:
        dff=df_split[(df_split.data.isin(data_sel))
            &(df_split.method.isin(method_sel))
                     &(df_split['rank'] ==rank_sel)
            &(df_split.criteria==criteria_sel)
           &(df_split.clustering == clus_sel)
                    ] 
    else:
            
 
        dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.noise ==noise_sel)
            &(df.sigma ==float(sigma_sel))
            &(df['rank'] ==rank_sel)
            &(df.criteria==criteria_sel)
           &(df.clustering == clus_sel)
              ] 
    this_palette_data=dict((i,palette_data[i]) for i in data_sel)
    this_palette=dict((i,palette[i]) for i in method_sel)
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        neww = new_data[(new_data.noise ==noise_sel)
                &(new_data['rank'] ==int(rank_sel))
                &(new_data.sigma ==float(sigma_sel))
               &(new_data.criteria==criteria_sel)]
        dff = pd.concat([dff, neww]) 
        
    corr1 = dff.groupby(['method'])[['Consistency','Accuracy']].corr(method = 'spearman').unstack().reset_index()    
#    corr = dff.groupby(['method'])[['Consistency','Accuracy']].corr().unstack().reset_index()    
    corr1.columns = [' '.join(col).strip() for col in corr1.columns.values]
    corr1=corr1[['method','Consistency Accuracy']]
    corr1 = sort(corr1,'method',list(this_palette.keys()))


    
    corr2 = dff.groupby(['data'])[['Consistency','Accuracy']].corr(method = 'spearman').unstack().reset_index()    
#    corr = dff.groupby(['method'])[['Consistency','Accuracy']].corr().unstack().reset_index()    
    corr2.columns = [' '.join(col).strip() for col in corr2.columns.values]
    corr2=corr2[['data','Consistency Accuracy']]
    corr2 = sort(corr2,'data',list(this_palette_data.keys()))
    fig1 = px.bar(corr1, x='method', y='Consistency Accuracy',
             range_y = [-1,1],
             color='method',color_discrete_map=this_palette,
             labels={'method':'Method', 'Consistency Accuracy':'Correlation'},
             title=" Rank Correlation between Accuracy and Consistency (aggregated over data)"
            )    
    fig2 = px.bar(corr2, x='data', y='Consistency Accuracy',
             range_y = [-1,1],
             color='data',color_discrete_map=this_palette_data,
             labels={'data':'Data', 'Consistency Accuracy':'Correlation'},
             title="Rank Correlation between Accuracy and Consistency (aggregated over methods)"
            )
    
    
    fig1.update_xaxes(tickangle=45)
    fig2.update_xaxes(tickangle=45)



    return fig2,fig1
                     


def build_line_raw_dr(data_sel, method_sel,
                 criteria_sel, noise_sel,sigma_sel,rank_sel,clus_sel,new_data=None):


    dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.noise ==noise_sel)
            &(df['rank'] ==rank_sel)
            &(df.criteria==criteria_sel)  
            &(df.clustering == clus_sel)]

    this_palette=dict((i,palette[i]) for i in method_sel)
    this_line_choice=dict((i,line_choice[i]) for i in method_sel)
    this_palette_data =  [i for i in palette_data.keys() if i in data_sel]   
    ###### input new data
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        neww = new_data[
            (new_data.noise ==noise_sel)&
             (new_data['rank'] ==rank_sel)&
            (new_data.criteria==criteria_sel)]
        dff = pd.concat([dff, neww]) 
        for mm in set(new_data['method']):
            this_palette[mm]='black'
            this_line_choice[mm]='solid'
            
    fig = px.line(dff,x="sigma", y='Consistency',color = 'method',
                            color_discrete_map=this_palette,
                                line_dash = 'method',
                      line_dash_map = this_line_choice,
                      labels={
                             "method": "Method"
                         },
                      facet_col="data",facet_col_wrap=3,facet_row_spacing=0.1,
#                   facet_col_spacing=0.1,
                  #width=1000, height=800,
            category_orders={'data':this_palette_data})
#     fig.update_xaxes(matches=None,showticklabels=True)
    fig.update_xaxes(showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_traces(line=dict(width=3))
    for i in range(1,5):
        fig.update_yaxes(title_text='Consistency',row=i, col=1,)
        for j in range(1,5):
            fig.update_xaxes(title_text='sigma',row=i, col=j,)
#             fig.update_yaxes(title_text='Accuracy',row=i, col=j,)

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
                
def build_scatter_raw_dr(data_sel, method_sel,
                 criteria_sel, noise_sel,sigma_sel,rank_sel,clus_sel,new_data=None):
    if noise_sel==None:
        dff=df_split[(df_split.data.isin(data_sel))
                        &(df_split['rank'] ==rank_sel)
            &(df_split.criteria==criteria_sel)
                       &(df_split.clustering == clus_sel)
            &(df_split.method.isin(method_sel))] 
    else:
        dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.noise ==noise_sel)
            &(df.sigma ==float(sigma_sel))
            &(df['rank'] ==rank_sel)
            &(df.criteria==criteria_sel)
                       &(df.clustering == clus_sel)]

    dff=dff.dropna()
    
    this_palette=dict((i,palette[i]) for i in method_sel)
    this_markers_choice=dict((i,markers_choice[i]) for i in method_sel)
    this_palette_data =  [i for i in palette_data.keys() if i in data_sel]   

    ###### input new data
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        neww = new_data[
            #(new_data.K ==k_sel)
             #   &
            (new_data.criteria==criteria_sel)]
        dff = pd.concat([dff, neww]) 
        for mm in set(new_data['method']):
            this_palette[mm]='black'
            this_markers_choice[mm]='star'
            
            
            
    fig = px.scatter(dff, x="Consistency", y="Accuracy", color='method', 
#                      trendline="ols",
               opacity=0.5,       facet_col="data",facet_col_wrap=3,facet_row_spacing=0.1,
                     #facet_col_spacing=0.1,
                     #width=1000, height=800,
                color_discrete_map=this_palette,
                symbol='method', symbol_map= this_markers_choice,
                 category_orders={"method":list(this_palette.keys())},
               labels=dict(Consistency='Consistency', method="Method"),

                )
   
   
    fig.update_traces(marker_size=15)
    fig.update_xaxes(showticklabels=True)
#     fig.update_xaxes(matches=None,showticklabels=True)
    for i in range(1,5):
        fig.update_yaxes(title_text='Accuracy',row=i, col=1,)
        for j in range(1,5):
            fig.update_xaxes(title_text='Consistency',row=i, col=j,)
#             fig.update_yaxes(title_text='Accuracy',row=i, col=j,)
    if new_data is not None:
        fig.add_trace(
        go.Scatter(
            x=neww['Consistency'],
            y=neww['Accuracy'],
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
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return fig
           
def build_heat_raw_dr(data_sel, method_sel,
                 criteria_sel, noise_sel,sigma_sel,rank_sel,clus_sel,new_data=None
                 ):
    

    if noise_sel==None:
        cross_ave = cross_split[(cross_split.data.isin(data_sel))
                &(cross_split['method1'].isin(method_sel))
                &(cross_split['method2'].isin(method_sel))
                &(cross_split.criteria==criteria_sel)
                &(cross['rank'] ==rank_sel)&(cross.clustering == clus_sel)]
        
    else:
        cross_ave = cross[(cross.data.isin(data_sel))
                &(cross['method1'].isin(method_sel))
                &(cross['method2'].isin(method_sel))
                &(cross.noise==noise_sel)
                &(cross.sigma==sigma_sel)
                &(cross.criteria==criteria_sel)
                &(cross['rank'] ==rank_sel)&(cross.clustering == clus_sel)]
        
        
    cross_ave=cross_ave.groupby(['data','method1','method2'],as_index=False)['value'].mean()
    subss = {}
    for i,dd in enumerate(data_sel):
        hh = cross_ave[cross_ave.data==dd].pivot("method1", "method2", "value")
        hh = hh.fillna(0)+hh.fillna(0).T
        np.fill_diagonal(hh.values, 1)
        hh=round(hh.reindex(columns=method_sel).reindex(method_sel),3)
        
        subss[dd]=hh
    this_palette=dict((i,palette[i]) for i in method_sel)
    
    fig = make_subplots(rows=len(data_sel)//3+1, cols=3, horizontal_spacing=0.05,
                    vertical_spacing=0.1,   subplot_titles=(data_sel))                                                                  

    for i,dd in enumerate(data_sel):
        bar1 = px.imshow(subss[dd],text_auto='.2f', origin='lower',)
        for trace in bar1.data:
            fig.add_trace(trace, i//3+1, i%3+1)
    for i in range(1,5):
        for j in range(2,4):
            fig.update_yaxes(showticklabels=False,row=i, col=j,)
    for i in range(1,5):
        fig.update_yaxes(title_text='Method',row=i, col=1)
        for j in range(1,4):
            fig.update_xaxes(title_text='Method',row=i, col=j,)
    fig.update_traces(coloraxis='coloraxis1',selector=dict(xaxis='x'))
    fig.update_layout(
        coloraxis=dict(colorscale= [(0,'whitesmoke'),(0.33, sns.xkcd_rgb['light lavender']),(0.66, sns.xkcd_rgb['lavender']),(1,sns.xkcd_rgb['amethyst'])],showscale = False),)
    fig.update_xaxes(tickangle=45)
    

    return fig


def build_dot_dr(data_sel, method_sel,
                 criteria_sel, noise_sel,sigma_sel,rank_sel,clus_sel, new_data=None
                 ):

    dff=df[(df.data.isin(data_sel))
             &(df['rank'] ==rank_sel)
           &(df.method.isin(method_sel))
            &(df.noise ==noise_sel)
            &(df.sigma ==float(sigma_sel))
            &(df.criteria==criteria_sel)
           &(df.clustering == clus_sel)]

    dff['size1']=(dff['Accuracy']**2)
    dff['size1']=[max(i,0.1) for i in dff['size1']]
    dff['size2']=(dff['Consistency']**2)
    dff['size2']=[max(i,0.1) for i in dff['size2']]    
    dff=dff.dropna()
    ###### input new data
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        neww = new_data[
            (new_data.noise ==noise_sel)
            &(new_data['rank'] ==rank_sel)
            &(new_data.sigma ==float(sigma_sel))
            &(new_data.criteria==criteria_sel)]
        dff = pd.concat([dff, neww]) 
        for mm in set(new_data['method']):
            this_palette[mm]='black'
            this_markers_choice[mm]='star'
    this_palette_data=dict((i,palette_data[i]) for i in data_sel)
    this_markers_choice=dict((i,markers_choice[i]) for i in method_sel)
    this_palette =  [i for i in palette.keys() if i in method_sel]   
           

    fig1 = px.scatter(dff, x="method", y="Consistency", color='data', 
                        size='size1',
                    color_discrete_map=this_palette_data,
                    #symbol='method', symbol_map= this_markers_choice,
              category_orders={"method":this_palette},
                   labels=dict( method="Method"),


               #  facet_col="acc_group",facet_col_wrap=3,
                    custom_data=['Accuracy','data'],
                    )
    fig1.update_traces(
            hovertemplate="<br>".join([
            "Data: %{customdata[1]}",
            "Method: %{x}",
            "Accuracy: %{customdata[0]}",
            "Consistency: %{y}",
                ]))   
    fig1.update_traces(line=dict(width=3))
    fig1.update_xaxes(matches=None)            
    

    fig2 = px.scatter(dff, x="method", y="Accuracy", color='data', 
                        size='size2',
                    color_discrete_map=this_palette_data,
                    #symbol='method', symbol_map= this_markers_choice,
              category_orders={"method":this_palette},
                   labels=dict(Consistency=criteria_sel, method="Method"),


               #  facet_col="acc_group",facet_col_wrap=3,
                    custom_data=['Consistency','data'],
                    )
    
    fig2.update_traces(
            hovertemplate="<br>".join([
            "Data: %{customdata[1]}",
            "Method: %{x}",
            "Accuracy: %{y}",
            "Consistency: %{customdata[0]}",
                ]))   
    fig2.update_traces(line=dict(width=3))
    fig2.update_xaxes(matches=None)
    return fig2,fig1





def build_heat_summary_dr(data_sel,method_sel,criteria_sel,noise_sel,sigma_sel,rank_sel,clus_sel
                 ):
    if noise_sel!=None:

        cross_ave = cross[cross.data.isin(data_sel)&(cross['clustering'] == clus_sel)&(cross['rank']==rank_sel)
                        & (cross['sigma']==sigma_sel)&(cross['criteria']==criteria_sel)&(cross['noise']==noise_sel)]

    else:
        cross_ave = cross_split[cross_split.data.isin(data_sel)&(cross_split['clustering'] == clus_sel)&(cross_split['rank']==rank_sel)&(cross['criteria']==criteria_sel)]
    cross_ave=cross_ave.groupby(['method1','method2'],as_index=False)['value'].mean()
    sub = cross_ave[(cross_ave['method1'].isin(method_sel))&(cross_ave['method2'].isin(method_sel))]

    sub = sub.pivot("method1", "method2", "value")
    sub = sub.fillna(0)+sub.fillna(0).T
    np.fill_diagonal(sub.values, 1)
    sub=round(sub.reindex(columns=method_sel).reindex(method_sel),3)


    fig = px.imshow(sub, text_auto=True, aspect="auto",
                        color_continuous_scale=[(0,'whitesmoke'),(0.33, sns.xkcd_rgb['light lavender']),(0.66, sns.xkcd_rgb['lavender']),(1,sns.xkcd_rgb['amethyst'])],
                   origin='lower',
           labels=dict(x="Method", y="Method", color="Consistency"))
    fig.update_xaxes(tickangle=45)
    fig.layout.coloraxis.showscale = False
    return fig





