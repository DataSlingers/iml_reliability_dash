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


nav = Navbar()
# header = html.H3(
#     'Reliability of Feature Importance'
# )
# accs=pd.read_csv('feature_impo_accs_reg.csv')
# accs['test_acc']=np.exp(-accs['test_acc'])
# accs=accs[accs.model!='LASSO']
# accs['test_acc']=(accs['test_acc']-min(accs['test_acc']))/(max(accs['test_acc'])-min(accs['test_acc']))
df = pd.read_csv("feature_impo_reg.csv")
#msee =[1/float(i) for i in df['Accuracy']]
msee =np.array([float(i) for i in df['MSE']])
df['Accuracy']= np.exp(-msee)

cross = pd.read_csv('cross_reg.csv')
cross_pred=pd.read_csv('fi_cross_pred_reg.csv')
# cross_pred['Consistency']=1-(cross_pred['value']-min(cross_pred['value']))/max(cross_pred['value'])-min(cross_pred['value'])
cross_pred['Consistency']=np.exp(-cross_pred['value'])

puris=pd.read_csv('feature_impo_pur_reg.csv')



criteria_options = df['criteria'].unique().tolist()
k_options =df['K'].unique().tolist()
method_category_options = ['Selected','Model Specific','Model Agnostic','All']
plot_summary_options = {'heatmap':'Consistency heatmap across methods',
                        'line':'Consistency across data sets',
                        'bump':'Bump plot of the most consistent methods across data sets',
                        'dot':'Consistency/predictive accuracy vs. methods',
                        'fit':'Scatter plots of interpretation consistency, predictive consistency, and preditvie accuracy',
                       # 'cor': 'Correlation between consistency and predictive error (MSE)'
                       }
plot_raw_options = {'scatter_raw':'Consistency vs. number of features for all data sets',
                   'line_raw':'Consistency vs. predictive accuracy for all data sets',
                                   'acc_raw':'Predictive accuracy for all data sets',
    'heatmap_raw':'Consistency heatmap across methods for all data sets'}
markers_choice = {'LASSO':'0',
                    'SVM':'0',
                    'Ridge':'0',

                    'Tree':'0',

                    'XGB':'0'  ,
                    'RF':'0' ,
            'MLP':'0',
                    'deepLIFT (MLP)':"square",
                    'Integrated Gradients (MLP)':"square",
                    'Epsilon-LRP (MLP)':"square",
                    'Saliency Maps (MLP)':"square",
                    'Guided Backpropagation (MLP)':"square",

                    'Occlusion (MLP)' :"x",
                    'Permutation (Ridge)':"x",
                    'Permutation (RF)':"x" ,
                                   'Permutation (XGB)':"x",
                    'Permutation (MLP)':"x",
                    'Shapley Value (Ridge)':"x" ,
                    'Shapley Value (RF)':"x",
                  'Shapley Value (XGB)': "x",       
                    'Shapley Value (MLP)':"x",
 }
# palette  = {
#             'LASSO':"purple",
#             'Ridge': 'indigo',
#             'SVM':"firebrick",           
#             'Tree':'deeppink',
#             'RF':"magenta",
#             'XGB':"violet",
#             'MLP':'green',
#             'deepLIFT (MLP)':"powderblue",
#             'Integrated Gradients (MLP)':'cornflowerblue',
#             'Epsilon-LRP (MLP)':"cyan",
#             'Guided Backpropagation (MLP)':"limegreen",           
#             'Saliency Maps (MLP)':"green",
#             'Occlusion (MLP)' :'greenyellow'  ,
#             'permutation (Ridge)':'tomato',
#             'permutation (RF)':"gold",
#             'permutation (XGB)':"olive",
#             'permutation (MLP)':"orange",
#              'Shapley Value (Ridge)':'chocolate',
#             'Shapley Value (RF)': "yellow",           
#             'Shapley Value (XGB)': "darkkhaki",       
#             'Shapley Value (MLP)':'peru',

#            }
# palette  = {
#     ## purple 
#         'Ridge': 'indigo',
#         'LASSO':"purple",
#         'SVM':"firebrick",           
#         'Tree':'deeppink',
#         'RF':"magenta",
#         'XGB':"violet",
#         ## green 
#         'MLP':"green",
#         'Epsilon-LRP (MLP)':"green",
#         'Guided Backpropagation (MLP)':"greenyellow",  

#         'deepLIFT (MLP)':"darkcyan",
#         'Integrated Gradients (MLP)':'medianseagreen',    
#         'Saliency Maps (MLP)':"olivedrab",
#         'Occlusion (MLP)' :'limegreen'  ,


#         ## orange scheme
#         'Permutation (Ridge)':'tomato',
#         'Permutation (RF)':"gold",
#          'Permutation (XGB)':"peru", 
#             'Permutation (MLP)':"yellow",       
#         ## blue 
#         'Shapley Value (Ridge)':'powderblue',
#         'Shapley Value (RF)': "cornflowerblue",           
#         'Shapley Value (XGB)': "skyblue",       
#         'Shapley Value (MLP)':'slateblue',
# }

palette  = {
    ## purple 
        'SVM':"deeppink",           
        'LASSO':"tomato",
        'Ridge': 'indigo',
        'Permutation (Ridge)':'purple',
       'Shapley Value (Ridge)':'firebrick',
        'Tree':'skyblue',
    
    ## blue 
        'RF':"slateblue",
       'Permutation (RF)':"darkturquoise",
               'Shapley Value (RF)': "cornflowerblue",           
    'XGB':"violet",
         'Permutation (XGB)':"peru", 
   'Shapley Value (XGB)': "magenta",       
    ## green 
       'MLP':"green",
        'Epsilon-LRP (MLP)':"green",
        'Guided Backpropagation (MLP)':"olivedrab",  
            'Permutation (MLP)':"orange",       
               'Shapley Value (MLP)':'gold',
        'deepLIFT (MLP)':"darkcyan",
        'Integrated Gradients (MLP)':'seagreen',    
        'Saliency Maps (MLP)':"greenyellow",
        'Occlusion (MLP)' :'limegreen'  ,
}


line_choice = {'LASSO':'solid',
               'Ridge':'solid',
                 'SVM':'solid',
              'Tree':'dot',
                  'RF':'dot' ,
                   'XGB':'dot',
                          'MLP':'dash',
                 'deepLIFT (MLP)':'dash',
               'Integrated Gradients (MLP)':'dash',
               'Epsilon-LRP (MLP)':'dash',
            'Permutation (MLP)':'dashdot',
             'Shapley Value (MLP)':'dashdot',
              
                        
            'Guided Backpropagation (MLP)':"dash",           
            'Saliency Maps (MLP)':"dash",
            'Occlusion (MLP)' :'dash'  ,
            'Permutation (Ridge)':'dashdot',
            'Permutation (RF)':"dashdot",
            'Shapley Value (Ridge)':'dashdot',
            'Shapley Value (RF)': "dashdot",       
            'Shapley Value (XGB)': "dashdot",       
            'Permutation (XGB)':"dashdot",

              }    

palette_data = {
         'News':"powderblue",   #671
    'Blog':"deepskyblue",  #187.13
    'Satellite': 'cornflowerblue' ,# 178 
    'Star':'slateblue',#55  
    'Communities' :"gold",#20   '
    'Bike':"orange",  
    'CPU':'salmon',  
    'Wine':'lightseagreen',  
    'Music':'olivedrab',    
    'Residential':'hotpink', #3.6  
    'Tecator':'firebrick',#1.9 
    'Word':"indigo",#0.9'
    'Riboflavin':"purple",
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

# plot_summary_options = ['heatmap','line','bump','dot','cor']
# plot_raw_options = ['scatter_raw','line_raw','heatmap_raw']


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
            
            ###############################
            ###############################            
            
            html.Hr(),
            html.P("Select: Consistency Metric"),
            dcc.RadioItems(
                id="criteria-select",
                options=[{"label": i, "value": i} for i in criteria_options],
                value='RBO',
            ),            
            html.Hr(),
            html.P("Select: Top K Features"),
            dcc.Dropdown(
                id="k-select",
                options=[{"label": i, "value": i} for i in k_options],
                
                value=10,
            ),
                
            html.Hr(),

            html.P("Select: Interpretability Method"),
            dcc.Dropdown(
                id="method-select",
                options=[{"label": i, "value": i} for i in meths],
                value=meths[1],
                multi=True,
            ),
            html.Hr(),
            html.P("Select: Interpretability Method Category"),
            dcc.RadioItems(
                id="method-select_c",
                options=[{"label": i, "value": i} for i in method_category_options],
                value=method_category_options[0],
            
            ),
            
            
            html.Hr(),
            html.P("Select: Data Sets"),
            dcc.Dropdown(
                id="data-select",
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



# options = [{'label':x.replace(', Illinois', ''), 'value': x} for x in df.columns]

# dropdown = html.Div(dcc.Dropdown(
#     id = 'pop_dropdown',
#     options = options,
#     value = 'Abingdon city, Illinois'
# ))

# output = html.Div(id = 'output',
#                 children = [],
#                 )

def App1_2():
    layout = html.Div([
        nav,
        dbc.Container([
   html.H1('Feature Importance (Regression)'),

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
   #         html.Div(id='show_cor'),
            ######### raw plots 
            html.Div(id='title_summary_raw'),
            html.Div(id='show_line_raw'),
            html.Div(id='show_scatter_raw'),
#                   html.Div(id='show_acc_raw'),
      html.Div(id='show_heatmap_raw'),
            
           
        ], 
          width={"size": 7, "offset": 1},

        
        )])])
        
    ])
    return layout

def build_scatter_reg(data_sel, method_sel,
                 k_sel, criteria_sel
                 ):
    this_palette=dict((i,palette[i]) for i in method_sel)
    this_markers_choice=dict((i,markers_choice[i]) for i in method_sel)
    
    
    dff=df[(df.data.isin(data_sel))
                &(df.method.isin(method_sel))
                &(df.K ==k_sel)
                &(df.criteria==criteria_sel)]
    fig = px.scatter(dff, x="Accuracy", y="Consistency", color='method', 
                 facet_col='data',
                 facet_col_wrap=3, 
                color_discrete_map=(palette),
                symbol='method', symbol_map= this_markers_choice,
                 text = 'method',
                 category_orders={"method":list(this_palette.keys())},
               labels=dict(Consistency=criteria_sel, method="Method",
                          Accuracy='Accuracy')

                )

    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-.2,
        xanchor="right",
        x=1
        ))
    fig.update_traces(    
        
       textposition='top right' 
)
    return fig

def build_bump_reg(data_sel, method_sel,
                 k_sel, criteria_sel,new_data=None
                 ):
####### filter data
    dff=df[(df.data.isin(data_sel))
                &(df.method.isin(method_sel))
                &(df.K ==k_sel)
                &(df.criteria==criteria_sel)]
    this_palette=dict((i,palette[i]) for i in method_sel)
    this_markers_choice=dict((i,markers_choice[i]) for i in method_sel)
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        neww = new_data[(new_data.K ==k_sel)
                &(new_data.criteria==criteria_sel)]
        dff = pd.concat([dff, neww],join='inner', ignore_index=True) 
        for mm in set(new_data['method']):
            this_palette[mm]='black'
        for mm in set(new_data['data']):
            data_sel = data_sel+[mm]
##### bump plot 
    df_ave = dff.groupby(['method','K','criteria'],as_index=False).mean()
    df_ave['data']='Average'
    df_ave=df_ave[['data','method','K','criteria','Consistency','Accuracy']]
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
                  labels=dict(data="Data",ranking='Rank',
                          Accuracy='Accuracy')
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

#         fig.update_layout(margin=dict( r=150))


    fig.add_annotation(dict(font=dict(color="grey",size=12),
                        x=-0.05, y=-0.1, 
                        text="Large N",
                        xref='paper',
                        yref='paper', 
                        showarrow=False))
    fig.add_annotation(dict(font=dict(color="grey",size=12),
                        x=1.1, y=-0.1, 
                        text="Large P",
                        xref='paper',
                        yref='paper', 
                        showarrow=False))
    return fig  

        
def build_acc_bar_reg(data_sel, method_sel,
                 k_sel, new_data=None):
    this_palette=dict((i,palette[i]) for i in method_sel)
    dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.K ==k_sel)]

    dff = dff.groupby(['method']).mean().reset_index()
    fig = px.bar(dff, x='method', y='Accuracy',
                 range_y = [0,1],
                 color='method',text_auto='.3',
                 color_discrete_map=this_palette,
                 labels=dict(method="Method",
                          Accuracy='Accuracy'),
                 title="Predictive Accuracy"
                )
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(height=300,showlegend=False)

    return fig


# def build_heat_summary_reg(data_sel,method_sel,
#                  k_sel, criteria_sel,new_data=None
#                  ):
#     cross_ave = cross[cross.data.isin(data_sel)]
#     cross_ave=cross_ave.groupby(['method1','method2','criteria','K'],as_index=False)['value'].mean()
#     sub = cross_ave[(cross_ave['method1'].isin(method_sel))&(cross_ave['method2'].isin(method_sel))]
#     sub = sub[(sub['K']==k_sel)&(sub['criteria']==criteria_sel)]
#     sub = sub.pivot("method1", "method2", "value")
#     sub = sub.fillna(0)+sub.fillna(0).T
#     np.fill_diagonal(sub.values, 1)
#     sub=round(sub.reindex(columns=method_sel).reindex(method_sel),3)


#     fig = px.imshow(sub, text_auto=True,color_continuous_scale='Purp',
#                     origin='lower',
#            labels=dict(x="Method", y="Method", color="Consistency"))
#     fig.layout.coloraxis.showscale = False
#     fig.update_xaxes(tickangle=45)

#     return fig
def build_heat_summary_reg(data_sel, method_sel,
                 k_sel, criteria_sel,new_data=None):
        cross_ave = cross[cross.data.isin(data_sel)]
        cross_ave=cross_ave.groupby(['method1','method2','criteria','K'],as_index=False)['value'].mean()
        sub = cross_ave[(cross_ave['method1'].isin(method_sel))&(cross_ave['method2'].isin(method_sel))]
        sub = sub[(sub['K']==k_sel)&(sub['criteria']==criteria_sel)]
        sub = sub.pivot("method1", "method2", "value")
        sub = sub.fillna(0)+sub.fillna(0).T
        np.fill_diagonal(sub.values, 1)
        sub=round(sub.reindex(columns=method_sel).reindex(method_sel),3)
        
        sub2 = cross_pred[(cross_pred.method1!='LASSO')&(cross_pred.method2!='LASSO')]
        sub2=sub2.groupby(['method1','method2'],as_index=False)['Consistency'].mean()
        sub2 = sub2.pivot("method1", "method2", "Consistency")
        sub2 = sub2.fillna(0)+sub2.fillna(0).T
        np.fill_diagonal(sub2.values, 1)
        sub2=round(sub2,3)
        
        fig = make_subplots(rows=1, cols=2, column_widths=[0.6, 0.4], 
                            horizontal_spacing=0.15,
                        vertical_spacing=0.05,   subplot_titles=('Interpretation Consistency','Prediction Accuracy'))

        heat1 = px.imshow(sub, text_auto=True, aspect="auto",color_continuous_scale='Purp',
                                                        origin='lower',labels=dict(x="Method", y="Method", color="Consistency"))

        heat2 = px.imshow( sub2, text_auto=True, aspect="auto",color_continuous_scale='Purp',
                                                        origin='lower',      labels=dict(x="Method", y="Method", color="Consistency"))
        heat2.layout.height = 500
        heat2.layout.width = 500
        for trace in heat1.data:
            fig.add_trace(trace, 1, 1)
        for trace in heat2.data:
            fig.add_trace(trace, 1, 2)
        fig.update_layout(
                      coloraxis=dict(colorscale='Purp', 
                                     showscale = False),)
        fig.update_xaxes(tickangle=45)# for trace in bar1.data:

        
        
        return fig
    


def build_heat_consis_reg(data_sel, method_sel,
                 k_sel, criteria_sel,new_data=None):

    dff=df[(df.data.isin(data_sel))
                &(df.method.isin(method_sel))
                &(df.K ==k_sel)
                &(df.criteria==criteria_sel)]

    this_palette=dict((i,palette[i]) for i in method_sel)
    this_line_choice=dict((i,line_choice[i]) for i in method_sel) 
    this_palette_data=dict((i,palette_data[i]) for i in data_sel)
    
    ###### input new data
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        neww = new_data[(new_data.K ==k_sel)
                &(new_data.criteria==criteria_sel)]
        dff = pd.concat([dff, neww]) 
        for mm in set(new_data['method']):
            this_palette[mm]='black'
            this_line_choice[mm]='solid'
            method_sel=method_sel+[mm]
        for mm in set(new_data['data']):
            this_palette_data[mm]='black'

            

    sub = dff.pivot("data", "method", "Consistency")
    sub=round(sub,3)
    sub= pd.DataFrame(sub, index=this_palette_data)
    sub=sub[method_sel]
    h = px.imshow(sub, text_auto=True, aspect="auto",range_color=(0,1),
                             color_continuous_scale=[(0, "seashell"),(0.7, "peachpuff"),(1, "darkorange")],
                  origin='lower',labels=dict(x="Method", y="Data", color="Consistency"))

    h.update_layout({
    'plot_bgcolor':'rgba(0, 0, 0, 0)',
    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    h.layout.height = 500
    h.layout.width = 1000
    
    
    dff_ac = dff[["data", "model", "Accuracy"]].drop_duplicates()
    sub2 = dff_ac.pivot("data", "model", "Accuracy")
    sub2=round(sub2,3)
    sub2= pd.DataFrame(sub2, index=this_palette_data)
    sub2=sub2[['Ridge','SVM','Tree','RF','XGB','MLP']]
    h2=px.imshow(sub2, text_auto=True, aspect="auto",
                 color_continuous_scale=[(0, "seashell"),(0.7, "peachpuff"),(1, "darkorange")],
                 range_color=(0,1),
                 origin='lower',labels=dict(x="Method", y="Data", color="Consistency"))
    h2.layout.height = 500
    h2.layout.width = 700

    fig= make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], 
                                horizontal_spacing=0.15,
                            vertical_spacing=0.05,   subplot_titles=('Interpretation Consistency','Prediction Accuracy'))

    for trace in h.data:
        fig.add_trace(trace, 1, 1)
    for trace in h2.data:
        fig.add_trace(trace, 1, 2)
    fig.update_xaxes(tickangle=45)# for trace in bar1.data:
    fig.update_layout(                             
                  coloraxis=dict(colorscale=[(0, "seashell"),(0.7, "peachpuff"),(1, "darkorange")],
                                 showscale = False),)
    
    return fig

def build_line_reg(data_sel, method_sel,
                 k_sel, criteria_sel,new_data=None
                 ):

    dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.K ==k_sel)
            &(df.criteria==criteria_sel)]
    method_sel = method_sel+['MLP']
    
    this_palette=dict((i,palette[i]) for i in method_sel)
    this_line_choice=dict((i,line_choice[i]) for i in method_sel)
    this_palette_data=dict((i,palette_data[i]) for i in data_sel)
    ###### input new data
    if new_data is not None:
        
        new_data = pd.DataFrame(new_data)
        new_data.replace({'normal':'Normal','laplace':'Laplace'})
        neww = new_data[(new_data.K ==k_sel)
                &(new_data.criteria==criteria_sel)]
        
        dff = pd.concat([dff, neww],join='inner', ignore_index=True) 
        
        
        for mm in set(new_data['method']):
            this_palette[mm]='black'
            this_line_choice[mm]='dash'
        for mm in set(new_data['data']):
            this_palette_data[mm]='black'


            


    fig1 = px.line(dff,x="data", y='Consistency',color = 'method',markers=True,

                        color_discrete_map=this_palette,
#                             line_dash = 'method',
#                   line_dash_map = this_line_choice,
                  labels=dict(Consistency=criteria_sel, data=  "Data",
                                      method="Method"),
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
    fig2 = px.line(dff[dff.model!='LASSO'],x="data", y='Accuracy',color = 'model',markers=True,

                            color_discrete_map=this_palette,
#                       line_dash_map = this_line_choice,
                      labels={
                             "model": "Method",'data':'Data'
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
        if fig2['data'][i]['name']!='MLP':
            fig2['data'][i]['showlegend']=False
        
    fig= make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5], 
                                horizontal_spacing=0.15,
                            vertical_spacing=0.05,   subplot_titles=('Interpretation Consistency','Prediction Accuracy'))
    for trace in fig1.data:
        fig.add_trace(trace, 1, 1)
    for trace in fig2.data:
        fig.add_trace(trace, 1, 2)
    fig.update_traces(line=dict(width=3))
    fig.add_annotation(dict(font=dict(color="grey",size=12),
                        x=-0.05, y=-0.1, 
                        text="Large N",
                        xref='paper',
                        yref='paper', 
                        showarrow=False))
    fig.add_annotation(dict(font=dict(color="grey",size=12),
                        x=0.55, y=-0.1, 
                        text="Large P",
                        xref='paper',
                        yref='paper', 
                        showarrow=False))
    fig.update_xaxes(categoryorder='array', categoryarray= datas)

    fig.update_xaxes(tickangle=45)

    return fig 


# def build_fit_reg(data_sel, method_sel,
#                  k_sel, criteria_sel,new_data=None
#                  ):

 
#     dff=df[(df.data.isin(data_sel))
#             &(df.method.isin(method_sel))
#             &(df.K ==k_sel)
#             &(df.criteria==criteria_sel)]
#     this_palette=dict((i,palette[i]) for i in method_sel)
#     this_markers_choice=dict((i,markers_choice[i]) for i in method_sel)
#     this_palette_data=dict((i,palette_data[i]) for i in data_sel)
#     ###### input new data
#     if new_data is not None:
#         new_data = pd.DataFrame(new_data)
#         neww = new_data[(new_data.K ==k_sel)
#                 &(new_data.criteria==criteria_sel)]
#         dff = pd.concat([dff, neww]) 
#         for mm in set(new_data['method']):
#             this_palette[mm]='black'
#             this_markers_choice[mm]='star'

            
#     fig1 = px.scatter(dff, x="Consistency", y="MSE", color='data', 
#                      trendline="ols",
#                 color_discrete_map=this_palette_data,
#                  category_orders={"Data":list(this_palette_data.keys())},
#                labels=dict(Consistency='Consistency', data="Data"),
#                 custom_data=['data','method'],
#                 )            
#     fig1.update_traces(
#         hovertemplate="<br>".join([
#         "Data: %{customdata[0]}",
#         "Method: %{customdata[1]}",
#         "Accuracy: %{y}",
#         "Consistency: %{x}",
#             ]))   
#     fig1.update_traces(line=dict(width=3),marker = dict(size=10),opacity=0.9)
    
#     region_lst = []


#     for trace in fig1["data"]:
#         trace["name"] = trace["name"].split(",")[0]

#         if trace["name"] not in region_lst and trace["marker"]['symbol'] == 'circle':
#             trace["showlegend"] = True
#             region_lst.append(trace["name"])
#         else:
#             trace["showlegend"] = False
#     model = px.get_trendline_results(fig1)
#     for i in range(0,len(fig1['data']),2):
#         fig1["data"][i+1]['customdata'] = fig1["data"][i]['customdata']
#         order = np.argsort(fig1["data"][i]['x'])
#         fig1["data"][i]['x'] = fig1["data"][i]['x'][order]
#         fig1["data"][i]['y'] = fig1["data"][i]['y'][order]


        
#         results = model.iloc[i//2]["px_fit_results"]
#         alpha = results.params[0]
#         beta = results.params[1]
#         p_beta = results.pvalues[1]
#         r_squared = results.rsquared

#         line1 = 'y = ' + str(round(alpha, 4)) + ' + ' + str(round(beta, 4))+'x'
#         line2 = 'p-value = ' + '{:.5f}'.format(p_beta)
#         line3 = 'R^2 = ' + str(round(r_squared, 3))
#         # summary = line1 + '<br>' + line2 + '<br>' + line3
#         fitted = np.repeat([[line1,line2,line3]], len(fig1["data"][i+1]['x']), axis=0)
#         fig1["data"][i+1]['customdata']=np.column_stack((fig1["data"][i+1]['customdata'],fitted))
#         fig1["data"][i+1]['hovertemplate'] = 'Data: %{customdata[0]}<br>Method: %{customdata[1]}<br> %{customdata[2]} <br> %{customdata[3]}<br> %{customdata[4]}'
#     if new_data is not None:
#         fig1.add_trace(
#         go.Scatter(
#             x=neww['MSE'],
#             y=neww['Consistency'],
#             mode='markers',
#             marker=dict(
#                 color=[this_palette[i] for i in neww['method']],
#                 symbol=[this_markers_choice[i] for i in neww['method']], 
#                 size=20
#             ),
#             showlegend=False,
#             hoverinfo='none'
#         )
#         )
        
#     fig2 = px.scatter(dff, x="Consistency", y="MSE", color='method', 
#                      trendline="ols",
#                 color_discrete_map=this_palette_data,
#                  category_orders={"Method":list(this_palette.keys())},
#                labels=dict(Consistency='Consistency', method="Method"),
#                 custom_data=['data','method'],
#                 )            
        
#     fig2.update_traces(
#         hovertemplate="<br>".join([
#         "Data: %{customdata[0]}",
#         "Method: %{customdata[1]}",
#         "Accuracy: %{y}",
#         "Consistency: %{x}",
#             ]))   
#     fig2.update_traces(line=dict(width=3),marker = dict(size=10),opacity=0.9)
    
#     region_lst2 = []


#     for trace in fig2["data"]:
#         trace["name"] = trace["name"].split(",")[0]

#         if trace["name"] not in region_lst2 and trace["marker"]['symbol'] == 'circle':
#             trace["showlegend"] = True
#             region_lst2.append(trace["name"])
#         else:
#             trace["showlegend"] = False
#     model2 = px.get_trendline_results(fig2)
#     for i in range(0,len(fig2['data']),2):
#         fig2["data"][i+1]['customdata'] = fig2["data"][i]['customdata']
#         order = np.argsort(fig2["data"][i]['x'])
#         fig2["data"][i]['x'] = fig2["data"][i]['x'][order]
#         fig2["data"][i]['y'] = fig2["data"][i]['y'][order]


        
#         results2 = model2.iloc[i//2]["px_fit_results"]
#         alpha = results2.params[0]
#         beta = results2.params[1]
#         p_beta = results2.pvalues[1]
#         r_squared = results2.rsquared

#         line1 = 'y = ' + str(round(alpha, 4)) + ' + ' + str(round(beta, 4))+'x'
#         line2 = 'p-value = ' + '{:.5f}'.format(p_beta)
#         line3 = 'R^2 = ' + str(round(r_squared, 3))
#         # summary = line1 + '<br>' + line2 + '<br>' + line3
#         fitted = np.repeat([[line1,line2,line3]], len(fig2["data"][i+1]['x']), axis=0)
#         fig2["data"][i+1]['customdata']=np.column_stack((fig2["data"][i+1]['customdata'],fitted))
#         fig2["data"][i+1]['hovertemplate'] = 'Data: %{customdata[0]}<br>Method: %{customdata[1]}<br> %{customdata[2]} <br> %{customdata[3]}<br> %{customdata[4]}'

#     if new_data is not None:
#         fig2.add_trace(
#         go.Scatter(
#             x=neww['MSE'],
#             y=neww['Consistency'],
#             mode='markers',
#             marker=dict(
#                 color=[this_palette[i] for i in neww['method']],
#                 symbol=[this_markers_choice[i] for i in neww['method']], 
#                 size=20
#             ),
#             showlegend=False,
#             hoverinfo='none'
#         )
#         )        
        

    
#     return fig1,fig2
def build_fit_reg(data_sel, method_sel,
                 k_sel, criteria_sel,new_data=None):

 
    dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.K ==k_sel)
            &(df.criteria==criteria_sel)]
    
    dff = pd.merge(dff, puris.groupby(['data','model']).mean().reset_index(), on = ['data','model'])    
    dff=sort(dff,'data',list(palette_data.keys()),'method',list(palette.keys()))

    this_palette=dict((i,palette[i]) for i in method_sel)
    this_markers_choice=dict((i,markers_choice[i]) for i in method_sel)
#     this_palette_data =  [i for i in palette_data.keys() if i in data_sel]   
    this_palette_data=dict((i,palette_data[i]) for i in data_sel)

    ###### input new data
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        neww = new_data[(new_data.K ==k_sel)
                &(new_data.criteria==criteria_sel)]
        dff = pd.concat([dff, neww]) 
        for mm in set(new_data['method']):
            this_palette[mm]='black'
            this_markers_choice[mm]='star'
            

    def get_scatter(df,x,y,color,palette,color_name,custom):
        f = px.scatter(df, x=x, y=y, color=color, 
                         trendline="ols",
                    color_discrete_map=palette,
                     category_orders={color_name:list(palette.keys())},
                    custom_data=[color,custom],
                    )  

        if x == 'Consistency' and y=="Accuracy":
            f.update_traces(
                hovertemplate="<br>".join([
                "Data: %{customdata[0]}",
                "Method: %{customdata[1]}",
                "Prediction Accuracy: %{y}",
                "Interpretation Consistency: %{x}",
                    ])) 
        if x == 'Consistency' and y=="Purity":
            f.update_traces(
                hovertemplate="<br>".join([
                "Data: %{customdata[0]}",
                "Method: %{customdata[1]}",
                "Prediction Consistency: %{y}",
                "Interpretation Consistency: %{x}",
                    ])) 
        if x == 'Accuracy' and y=="Purity":
            f.update_traces(
                hovertemplate="<br>".join([
                "Data: %{customdata[0]}",
                "Method: %{customdata[1]}",
                "Prediction Consistency: %{y}",
                "Prediction Accuracy: %{x}",
                    ])) 
        f.update_layout(xaxis_range=[0,1])
        model = px.get_trendline_results(f)
        nn=len(f['data'])//2
        pvs = pd.DataFrame(columns = ['data','p-values'])
        for i in range(0,len(f['data']),2):
            f["data"][i+1]['customdata'] = f["data"][i]['customdata']
            order = np.argsort(f["data"][i]['x'])
            f["data"][i]['x'] = f["data"][i]['x'][order]
            f["data"][i]['y'] = f["data"][i]['y'][order]



            results = model.iloc[i//2]["px_fit_results"]
            alpha = results.params[0]
            beta = results.params[1]
            p_beta = results.pvalues[1]
            r_squared = results.rsquared

            line1 = 'y = ' + str(round(alpha, 4)) + ' + ' + str(round(beta, 4))+'x'
            line2 = 'p-value = ' + '{:.5f}'.format(p_beta)
            line3 = 'R^2 = ' + str(round(r_squared, 3))
            # summary = line1 + '<br>' + line2 + '<br>' + line3
            fitted = np.repeat([[line1,line2,line3]], len(f["data"][i+1]['x']), axis=0)
            f["data"][i+1]['customdata']=np.column_stack((f["data"][i+1]['customdata'],fitted))
            f["data"][i+1]['hovertemplate'] = 'Data: %{customdata[0]}<br>Method: %{customdata[1]}<br> %{customdata[2]} <br> %{customdata[3]}<br> %{customdata[4]}'
            if beta<0 and round(p_beta*nn,5)<0.05:
                pvs.loc[len(pvs)]=[f["data"][i+1]['name'],str(min(round(p_beta*nn,5),1))+ ' (Negative)']
            else:
                pvs.loc[len(pvs)]=[f["data"][i+1]['name'],min(round(p_beta*nn,5),1)]
        return f,pvs
    
    
    fig1,pv1 = get_scatter(dff,'Consistency',y="Accuracy",color='data',
                palette=this_palette_data,color_name='Data',custom='method')
    fig2,pv2 = get_scatter(dff, x="Consistency", y="Purity", color='data', 
                    palette=this_palette_data,color_name='Data',custom='method')

    fig3,pv3 = get_scatter(dff, x="Accuracy", y="Purity", color='data', 
                    palette=this_palette_data,color_name='Data',custom='method')
    
    fig4,pv4= get_scatter(dff,'Consistency',y="Accuracy",color='method',
                palette=this_palette,color_name='Method',custom='data')
    
    fig5,pv5= get_scatter(dff, x="Consistency", y="Purity", color='method', 
                    palette=this_palette,color_name='Method',custom='data')

    fig6,pv6 = get_scatter(dff, x="Accuracy", y="Purity", color='method', 
                    palette=this_palette,color_name='Method',custom='data')
    
    
    
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.1,vertical_spacing=0.05)
                       #subplot_titles=('Interpretation Consistency vs. Prediction Accuracy',
#               'Interpretation Consistency vs.Prediction Consistency',
#              'Prediction Consistency vs.  Prediction Accuracy'))
    for trace in fig1.data:
        fig.add_trace(trace, 1, 1)
    for trace in fig2.data:
        fig.add_trace(trace, 1, 2)
    for trace in fig3.data:
        fig.add_trace(trace, 1, 3)
    region_lst = []


    for trace in fig["data"]:
        trace["name"] = trace["name"].split(",")[0]

        if trace["name"] not in region_lst and trace["marker"]['symbol'] == 'circle':
            trace["showlegend"] = True
            region_lst.append(trace["name"])
        else:
            trace["showlegend"] = False
    fig.update_xaxes(title_text="Interpretation Consistency", row=1, col=1,title_standoff = 0)
    fig.update_xaxes(title_text="Interpretation Consistency", row=1, col=2,title_standoff = 0)
    fig.update_xaxes(title_text="Prediction Accuracy", row=1, col=3,title_standoff = 0)
    fig.update_yaxes(title_text="Prediction Accuracy", row=1, col=1,title_standoff = 0)
    fig.update_yaxes(title_text="Prediction Consistency",  row=1, col=2,title_standoff = 0)
    fig.update_yaxes(title_text="Prediction Consistency",  row=1, col=3,title_standoff = 0)



    fig.update_traces(line=dict(width=3),marker = dict(size=7),opacity=0.8)
    
    
    figg = make_subplots(rows=1, cols=3, horizontal_spacing=0.1,vertical_spacing=0.05)
#                          subplot_titles=('Interpretation Consistency vs. Prediction Accuracy',
#               'Interpretation Consistency vs.Prediction Consistency',
#              'Prediction Consistency vs.  Prediction Accuracy'))
    for trace in fig4.data:
        figg.add_trace(trace, 1, 1)
    for trace in fig5.data:
        figg.add_trace(trace, 1, 2)
    for trace in fig6.data:
        figg.add_trace(trace, 1, 3)
    region_lst = []


    for trace in figg["data"]:
        trace["name"] = trace["name"].split(",")[0]

        if trace["name"] not in region_lst and trace["marker"]['symbol'] == 'circle':
            trace["showlegend"] = True
            region_lst.append(trace["name"])
        else:
            trace["showlegend"] = False
    figg.update_xaxes(title_text="Interpretation Consistency", row=1, col=1,title_standoff = 0)
    figg.update_xaxes(title_text="Interpretation Consistency", row=1, col=2,title_standoff = 0)
    figg.update_xaxes(title_text="Prediction Accuracy", row=1, col=3,title_standoff = 0)
    figg.update_yaxes(title_text="Prediction Accuracy", row=1, col=1,title_standoff = 0)
    figg.update_yaxes(title_text="Prediction Consistency",  row=1, col=2,title_standoff = 0)
    figg.update_yaxes(title_text="Prediction Consistency",  row=1, col=3,title_standoff = 0)



    figg.update_traces(line=dict(width=3),marker = dict(size=7),opacity=0.8)
    
    pv = pd.merge(pd.merge(pv1,pv2,on='data'),pv3,on='data')
#     pv=round(pv,3)

    
    pvv= pd.merge(pd.merge(pv4,pv5,on='data'),pv6,on='data')
#     pvv=round(pvv,3)
    fig_pv = go.Figure(data=[go.Table(header=dict(values=['Data','Interpretation Consistency vs. Prediction Accuracy',
              'Interpretation Consistency vs.Prediction Consistency',
             'Prediction Consistency vs.  Prediction Accuracy']),
                 cells=dict(values=np.array(pv.T)))
                     ])
    fig_pv.update_layout(title_text='P-values of fitted line (bonferroni corrected)')
    fig_pvv = go.Figure(data=[go.Table(header=dict(values=['Data','Interpretation Consistency vs. Prediction Accuracy',
              'Interpretation Consistency vs.Prediction Consistency',
             'Prediction Consistency vs.  Prediction Accuracy']),
                 cells=dict(values=np.array(pvv.T)))
                     ])

    fig_pvv.update_layout(title_text='P-values of fitted line (bonferroni corrected)')
    return fig,figg,fig_pv,fig_pvv
            
                    
            
def build_acc_raw_reg(data_sel,new_data=None):


#     accs=pd.read_csv('feature_impo_accs.csv')
    acc=accs[(accs.data.isin(data_sel)) ]

    fig = px.box(acc, x="model", y="test_acc", color='model', 
                     facet_col="data",facet_col_wrap=3,facet_row_spacing=0.1,
                        color_discrete_map =palette, 
               labels=dict(data='Data',test_acc='Accuracy', model="Method"))
   
    fig.update_traces(marker_size=10)
    fig.update_xaxes(matches=None,showticklabels=True)
    fig.update_yaxes(matches=None,showticklabels=True)
    return fig

def build_heat_raw_reg(data_sel, method_sel,
                 k_sel, criteria_sel,new_data=None):
    
    
    
    cross_ave = cross[(cross.data.isin(data_sel))
                &(cross['method1'].isin(method_sel))
                &(cross['method2'].isin(method_sel))
                &(cross.K ==k_sel)
                &(cross.criteria==criteria_sel)]
    cross_ave=cross_ave.groupby(['data','method1','method2','criteria','K'],as_index=False)['value'].mean()
    subss = {}
    for i,dd in enumerate(data_sel):
        hh = cross_ave[cross_ave.data==dd].pivot("method1", "method2", "value")
        hh = hh.fillna(0)+hh.fillna(0).T
        np.fill_diagonal(hh.values, 1)
        hh=round(hh.reindex(columns=method_sel).reindex(method_sel),3)
        
        subss[dd]=hh

#     tt =[[i]  for i in data_sel for _ in range(2)]
#     tt = [item for sublist in tt for item in sublist]
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

    fig.update_traces(coloraxis='coloraxis1',selector=dict(xaxis='x'))
    fig.update_layout(
                  coloraxis=dict(colorscale='Purp', 
                                 showscale = False),)
    fig.update_xaxes(tickangle=45)
#     fig['layout'].update(height=800, width=800)
    fig.for_each_yaxis(lambda xaxis: xaxis.update(showticklabels=True))
    return fig



    
#     fig.update_traces(
#         hovertemplate="<br>".join([
#         "Data: %{customdata[0]}",
#         "Method: %{customdata[1]}",
#         "MSE: %{x}",
#         "Consistency: %{y}",
#             ]))   
#     fig.update_traces(line=dict(width=3))

    
#     if new_data is not None:
#         fig.add_trace(
#         go.Scatter(
#             x=neww['MSE'],
#             y=neww['Consistency'],
#             mode='markers',
#             marker=dict(
#                 color=[this_palette[i] for i in neww['method']],
#                 symbol=[this_markers_choice[i] for i in neww['method']], 
#                 size=20
#             ),
#             showlegend=False,
#             hoverinfo='none'
#         )
#         )
        
#     return fig
        
def build_cor_reg(data_sel, method_sel,
                 k_sel, criteria_sel,new_data=None
                 ):

 
    dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.K ==k_sel)
            &(df.criteria==criteria_sel)]
    
    
    this_palette_data=dict((i,palette_data[i]) for i in data_sel)
    this_palette=dict((i,palette[i]) for i in method_sel)
    ###### input new data
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        neww = new_data[(new_data.K ==k_sel)
                &(new_data.criteria==criteria_sel)]
        dff = pd.concat([dff, neww]) 
        for mm in set(new_data['method']):
            this_palette[mm]='black'
            
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

#     fig2 = px.bar(corr2, x='data', y='Consistency MSE',
#              range_y = [-1,1],
#              color='data',color_discrete_map=this_palette_data,
#              labels={'data':'Data', 'Consistency MSE':'Correlation'},
#              title="Correlation between MSE and Consistency"
#             )
#     fig1 = px.bar(corr1, x='method', y='Consistency MSE',
#              range_y = [-1,1],
#              color='method',color_discrete_map=this_palette,
#              labels={'method':'Method', 'Consistency MSE':'Correlation'},
#              title="Correlation between MSE and Consistency"
#             )
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
               
def build_line_raw_reg(data_sel, method_sel,
                 k_sel, criteria_sel,new_data=None):

    dff=df[(df.data.isin(data_sel))
                &(df.method.isin(method_sel))
               # &(df.K ==k_sel)
                &(df.criteria==criteria_sel)]

    this_palette=dict((i,palette[i]) for i in method_sel)
    this_line_choice=dict((i,line_choice[i]) for i in method_sel)
    this_palette_data =  [i for i in palette_data.keys() if i in data_sel]   
    ###### input new data
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        neww = new_data[
            #(new_data.K ==k_sel)
            #    &
            (new_data.criteria==criteria_sel)]
        dff = pd.concat([dff, neww]) 
        for mm in set(new_data['method']):
            this_palette[mm]='black'
            this_line_choice[mm]='solid'
            
    fig = px.line(dff,x="K", y='Consistency',color = 'method',

                            color_discrete_map=this_palette,
                                line_dash = 'method',
                      line_dash_map = this_line_choice,
                      labels={
                             "method": "Method"
                         },
                      facet_col="data",facet_col_wrap=4,facet_row_spacing=0.05,
                  #width=1000, height=800,
            category_orders={'data':this_palette_data})
    fig.update_xaxes(matches=None,showticklabels=True)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
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
    fig.for_each_xaxis(lambda xaxis: xaxis.update(showticklabels=True))
        
    return fig
                
def build_scatter_raw_reg(data_sel, method_sel,
                 k_sel, criteria_sel,new_data=None):

 
    dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.K ==k_sel)
            &(df.criteria==criteria_sel)]
    
    
    this_palette=dict((i,palette[i]) for i in method_sel)
    this_markers_choice=dict((i,markers_choice[i]) for i in method_sel)
    ###### input new data
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        neww = new_data[
            (new_data.K ==k_sel)&
            (new_data.criteria==criteria_sel)]
        dff = pd.concat([dff, neww]) 
        for mm in set(new_data['method']):
            this_palette[mm]='black'
            this_markers_choice[mm]='star'
            
            
            
    fig = px.scatter(dff, x="Consistency", y="Accuracy", color='method', 
#                      trendline="ols",
                     opacity=0.5, facet_col="data",facet_col_wrap=4,
                     #width=1000, height=800,
                color_discrete_map=this_palette,
                symbol='method', symbol_map= this_markers_choice,
                 category_orders={"method":list(this_palette.keys())},
               labels=dict(Consistency=criteria_sel, method="Method"),

                )
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
    fig.update_traces(marker_size=10)
    fig.update_xaxes(matches=None,showticklabels=True)
    fig.for_each_xaxis(lambda xaxis: xaxis.update(showticklabels=True))

    return fig
             
# def build_heat_raw_reg(data_sel, method_sel,
#                  k_sel, criteria_sel,new_data=None):
    
    
    
#     cross_ave = cross[(cross.data.isin(data_sel))
#                 &(cross['method1'].isin(method_sel))
#                 &(cross['method2'].isin(method_sel))
#                 &(cross.K ==k_sel)
#                 &(cross.criteria==criteria_sel)]
#     cross_ave=cross_ave.groupby(['data','method1','method2','criteria','K'],as_index=False)['value'].mean()
# #     sub = cross_ave[(cross_ave['method1'].isin(method_sel))&(cross_ave['method2'].isin(method_sel))]
# #     sub = sub[(sub['K']==k_sel)&(sub['criteria']==criteria_sel)]
#     method_acc_sel=['LASSO','Ridge','SVM','Tree','XGB','RF','Occlusion (MLP)']
    
#     dff=df[(df.data.isin(data_sel))
#                 &(df.method.isin(method_acc_sel))
#                 &(df.K ==k_sel)
#                 &(df.criteria==criteria_sel)]
#     dff=dff.replace({'Occlusion (MLP)':'MLP'})
    
#     dff = dff.groupby(['method','data']).mean().reset_index()
#     subss = {}
#     for i,dd in enumerate(data_sel):
#         hh = cross_ave[cross_ave.data==dd].pivot("method1", "method2", "value")
#         hh = hh.fillna(0)+hh.fillna(0).T
#         np.fill_diagonal(hh.values, 1)
#         hh=round(hh.reindex(columns=method_sel).reindex(method_sel),3)
        
#         subss[dd]=hh

#     tt =[[i]  for i in data_sel for _ in range(2)]
#     tt = [item for sublist in tt for item in sublist]
#     this_palette=dict((i,palette[i]) for i in method_sel)

#     fig = make_subplots(rows=len(data_sel), cols=2, horizontal_spacing=0.05,
#                     vertical_spacing=0.05,                     
#                                      subplot_titles=(tt)                                                                  )

#     for i,dd in enumerate(data_sel):
#         bar1 = px.imshow(subss[dd],text_auto='.2f', origin='lower',)
#         bar2 = px.bar(dff[dff.data ==dd], x='method', y='MSE',range_y = [0,1],
#                         color_discrete_map =palette,color='method',
#                      text_auto='.3' )

#         for trace in bar1.data:
#             fig.add_trace(trace, i+1, 1)
#         for trace in bar2.data:
#             trace["width"] = 1
#             trace["showlegend"] = False

#             fig.add_trace(trace, i+1, 2)

#         fig.update_traces(coloraxis='coloraxis1',selector=dict(xaxis='x'))
#         fig.update_layout(
#                       coloraxis=dict(colorscale='Purp', 
#                                      showscale = False),)
#         fig.update_xaxes(tickangle=45)
#     fig['layout'].update(height=4000, width=800)
#     return fig

def build_dot_reg(data_sel, method_sel,
                 k_sel, criteria_sel,new_data=None):

 
    dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.K ==k_sel)
            &(df.criteria==criteria_sel)]
    this_palette_data=dict((i,palette_data[i]) for i in data_sel)
    this_palette=[i for i in palette.keys() if i in method_sel]
    this_markers_choice=dict((i,markers_choice[i]) for i in method_sel)

    dff['size1']=(dff['Accuracy']**2)
    dff['size1']=[max(i,0.1) for i in dff['size1']]
    dff['size2']=(dff['Consistency']**2)
    dff['size2']=[max(i,0.1) for i in dff['size2']]    
    ###### input new data
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        neww = new_data[(new_data.K ==k_sel)
                &(new_data.criteria==criteria_sel)]
        dff = pd.concat([dff, neww]) 
        for mm in set(new_data['method']):
            this_palette[mm]='black'
            this_markers_choice[mm]='star'
            

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





