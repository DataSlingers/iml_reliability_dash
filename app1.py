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
from dash import dash_table
from plotly.subplots import make_subplots



nav = Navbar()
# header = html.H3(
#     'Reliability of Feature Importance'
# )

df = pd.read_csv("feature_impo.csv")
# df=df.dropna()
cross = pd.read_csv('cross_fi.csv')


data_options = df['data'].unique().tolist()
method_options = df['method'].unique().tolist()
criteria_options = df['criteria'].unique().tolist()
method_category_options = ['All','Model Specific','Model Agnostic']

k_options =df['K'].unique().tolist()
plot_summary_options = {'heatmap':'Consistency heatmap across methods',
                        'line':'Consistency across data sets',
                        'bump':'Bump plot of the most consistent methods across data sets',
#                         'fit':'Consistency vs. predictive accuracy',
                        'dot':'Consistency/predictive accuracy vs. methods',
                       # 'cor': 'Correlation between onsistency and predictive accuracy'
                       }
plot_raw_options = {'scatter_raw':'Consistency vs. number of features for all data sets',
                   'line_raw':'Consistency vs. predictive accuracy for all data sets',
                    'heatmap_raw':'Consistency heatmap across methods for all data sets'}

markers_choice = {'LogisticLASSO':'circle',
                    'SVM':'circle',
                    'LogisticRidge':'circle',

                    'Tree':'circle',

                    'XGB':'circle'  ,
                    'RF':'circle' ,

                    'deepLIFT (MLP)':"square",
                    'Integrated Gradients (MLP)':"square",
                    'Epsilon-LRP (MLP)':"square",
                    'Saliency Maps (MLP)':"square",
                    'Guided Backpropagation (MLP)':"square",

                    'Occlusion (MLP)' :"x",
                    'permutation (LogisticRidge)':"x",
                    'permutation (RF)':"x" ,
                    'permutation (MLP)':"x",
                      'Shapley Value (XGB)': "x",       
                    'permutation (XGB)':"x",
                  'Shapley Value (LogisticRidge)':"x" ,
                    'Shapley Value (RF)':"x",
                    'Shapley Value (MLP)':"x"}
palette  = {
        'LogisticRidge': 'indigo',
        'LogisticLASSO':"purple",

        'SVM':"firebrick",           
        'Tree':'deeppink',
        'RF':"magenta",
        'XGB':"violet",
        'deepLIFT (MLP)':"powderblue",
        'Integrated Gradients (MLP)':'cornflowerblue',
        'Epsilon-LRP (MLP)':"cyan",
        'permutation (MLP)':"orange",       
        'Shapley Value (MLP)':'peru',


        'Guided Backpropagation (MLP)':"limegreen",           
        'Saliency Maps (MLP)':"green",
        'Occlusion (MLP)' :'greenyellow'  ,
        'permutation (LogisticRidge)':'tomato',
        'permutation (RF)':"gold",
        'Shapley Value (LogisticRidge)':'chocolate',
        'Shapley Value (RF)': "yellow",           
     'Shapley Value (XGB)': "darkkhaki",       
               'permutation (XGB)':"olive",      }

line_choice = {'LogisticLASSO':'solid',
               'LogisticRidge':'solid',
               'SVM':'solid',
                'Tree':'dot',
                'RF':'dot' ,
               'XGB':'dot',
               'deepLIFT (MLP)':'dash',
               'Integrated Gradients (MLP)':'dash',
               'Epsilon-LRP (MLP)':'dash',
            'permutation (MLP)':'dashdot',
             'Shapley Value (MLP)':'dashdot',
              
            'Guided Backpropagation (MLP)':"dash",           
            'Saliency Maps (MLP)':"dash",
            'Occlusion (MLP)' :'dash'  ,
            'permutation (LogisticRidge)':'dashdot',
            'permutation (RF)':"dashdot",
            'Shapley Value (LogisticRidge)':'dashdot',
            'Shapley Value (RF)': "dashdot",       
              
                     'Shapley Value (XGB)': "dashdot",       
            'permutation (XGB)':"dashdot",     }    

palette_data = {'Bean':"powderblue",
    'Call':'cornflowerblue',  
    'Statlog':'deepskyblue',      
    'Theorem':'slateblue',   
    'MNIST Digit':"cyan",   
    'Spam base':"green", 
    'Author':'yellow',           
    'Amphibians':'lightseagreen',  
    'Madelon' :'greenyellow', 
    'TCGA':'hotpink',
    'DNase':"firebrick",                   
    'Religion': 'indigo',
    'PANCAN':"purple",
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
df.K=[int(i) for i in df.K] 
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
            html.P("Select: Consistency Metric"),
            dcc.RadioItems(
                id="criteria-select",
                    
                options=[{"label": i, "value": i} for i in criteria_options],
                value=criteria_options[2],
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
                value=meths[0:10],
                multi=True,
            ),
            html.Br(),
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
                options=[{"label": i, "value": i} for i in data_options],
                value=data_options[:],
                multi=True,
            ),
            html.Br(),

                        
            #################################
            ########### select figures 
            #################################
            

            html.P("Select Summary Graphs you want to show"),
            dcc.Checklist(id="all_summary",
                          options=[{"label": 'All', "value":'All_summary' }],value= ['All_summary']),
            dcc.Checklist(id="select_summary",
                options=[{"label": plot_summary_options[i], "value": i} for i in plot_summary_options],
                value=list(plot_summary_options.keys()),
            ),        
            
            html.Hr(),
            html.P("Select Raw Graphs you want to show"),
            dcc.Checklist(id="all_raw",
                          options=[{"label": 'All', "value":'All_raw' }],value= ['All_raw']),
            dcc.Checklist(id="select_raw",
                options=[{"label": plot_raw_options[i], "value": i} for i in plot_raw_options],
                value=list(plot_raw_options.keys()),
            ),         
            
            
            html.Br(),

            dbc.Button('Submit', id='submit-button',n_clicks=1, color="primary",className="me-1"),
            dbc.Button('Reset',id='reset-button',n_clicks=0, color="secondary",className="me-1"),
            html.Hr(),
          
            ###############################
            ###############################        
        ],
    )





def App1():
    layout = html.Div([
        nav,
        dbc.Container([
   html.H1('Feature Importance (Classification)'),

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
            html.Div(id='show_line'),
            html.Div(id='show_bump'),
            html.Div(id='show_heatmap'),
            html.Div(id='show_fit'),
#             html.Div(id='show_dot'),
#            html.Div(id='show_cor'),
            ######### raw plots 
            html.Div(id='title_summary_raw'),
            html.Div(id='show_line_raw'),
            html.Div(id='show_scatter_raw'),
            html.Div(id='show_heatmap_raw'),

           
        ], 
          width={"size": 7, "offset": 1},

#                 width='auto'
                #,align='center'
                     )])])
           
      #  )])
     #  ],fluid=True)
        
    ])
    return layout

















def build_scatter(data_sel, method_sel,
                 k_sel, criteria_sel
                 ):
    dff=df[(df.data.isin(data_sel))
                &(df.method.isin(method_sel))
                &(df.K ==k_sel)
                &(df.criteria==criteria_sel)]
    this_palette=dict((i,palette[i]) for i in method_sel)
    this_markers_choice=dict((i,markers_choice[i]) for i in method_sel)
    
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
       textfont=dict(
                family="sans serif",
                size=10,
           #opacity=0.1,
                color="darkgrey"),
            marker=dict(size=20,
                        #opacity=0.1,
                        line=dict(width=2,
                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'),
     
)
    fig.update_xaxes(categoryorder='array', categoryarray= datas)

    return fig
# def show_data(new_data=None):
    
#     if new_data is None:
#         return df.to_dict('records'), [{"name": i, "id": i} for i in df.columns]
#     else:
#         return new_data.to_dict('records'), [{"name": i, "id": i} for i in df.columns]
    
    
    
def build_bump(data_sel, method_sel,
                 k_sel, criteria_sel,new_data=None):
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
        dff = pd.concat([dff, neww]) 
        for mm in set(neww['method']):
            this_palette[mm]='black'
        
    ##### bump plot 
    df_ave = dff.groupby(['method','K','criteria'],as_index=False).mean()
    df_ave['data']='Average'
    df_ave=df_ave[['data','method','Consistency']]
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
                  labels=dict(data="Data",ranking='Rank')
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

    fig.update_xaxes(categoryorder='array', categoryarray= datas)

    
    return fig  


def build_heat_summary(data_sel, method_sel,
                 k_sel, criteria_sel,new_data=None):
        cross_ave = cross[cross.data.isin(data_sel)]
        cross_ave=cross_ave.groupby(['method1','method2','criteria','K'],as_index=False)['value'].mean()
        sub = cross_ave[(cross_ave['method1'].isin(method_sel))&(cross_ave['method2'].isin(method_sel))]
        sub = sub[(sub['K']==k_sel)&(sub['criteria']==criteria_sel)]
        sub = sub.pivot("method1", "method2", "value")
        sub = sub.fillna(0)+sub.fillna(0).T
        np.fill_diagonal(sub.values, 1)
        sub=round(sub.reindex(columns=method_sel).reindex(method_sel),3)
        
        
        fig = px.imshow(sub, text_auto=True, aspect="auto",color_continuous_scale='Purp',
                                                origin='lower',
               labels=dict(x="Method", y="Method", color="Consistency"))
        fig.update_xaxes(tickangle=45)
        fig.layout.coloraxis.showscale = False
        return fig



def build_acc_bar(data_sel, method_sel,
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
                 labels={'method':'Method', 'Accuracy':'Predictive Accuracy'},
                 title="Predictive Accuracy"
                )
    fig.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
    fig.update_layout(height=300,showlegend=False)

    return fig



def build_line(data_sel, method_sel,
                 k_sel, criteria_sel,new_data=None):

    dff=df[(df.data.isin(data_sel))
                &(df.method.isin(method_sel))
                &(df.K ==k_sel)
                &(df.criteria==criteria_sel)]

    this_palette=dict((i,palette[i]) for i in method_sel)
    this_line_choice=dict((i,line_choice[i]) for i in method_sel)
    
    ###### input new data
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        neww = new_data[(new_data.K ==k_sel)
                &(new_data.criteria==criteria_sel)]
        dff = pd.concat([dff, neww]) 
        for mm in set(new_data['method']):
            this_palette[mm]='black'
            this_line_choice[mm]='solid'
            
    fig = px.line(dff,x="data", y='Consistency',color = 'method',markers=True,

                            color_discrete_map=this_palette,
                                line_dash = 'method',
                      line_dash_map = this_line_choice,
                      labels={
                             "method": "Method",'data':'Data'
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

    fig.update_xaxes(categoryorder='array', categoryarray= datas)
    return fig
            
        
def build_fit(data_sel, method_sel,
                 k_sel, criteria_sel,new_data=None):

 
    dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.K ==k_sel)
            &(df.criteria==criteria_sel)]
    
    
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
            
            
    fig = px.scatter(dff, x="Consistency", y="Accuracy", color='data', 
                     trendline="ols",
                color_discrete_map=this_palette_data,
#                 symbol='method', symbol_map= this_markers_choice,
                 category_orders={"Data":list(this_palette_data.keys())},
               labels=dict(Consistency='Consistency', data="Data"),

                custom_data=['data','method'],
                )            
#     fig = px.scatter(dff, x="Consistency", y="Accuracy", color='method', 
#                      trendline="ols",
#                 color_discrete_map=this_palette,
#                 symbol='method', symbol_map= this_markers_choice,
#                  category_orders={"method":list(this_palette.keys())},
#                labels=dict(Consistency=criteria_sel, method="Method"),

#                 custom_data=['data','method'],
#                 )
    region_lst = []


    for trace in fig["data"]:
        trace["name"] = trace["name"].split(",")[0]

        if trace["name"] not in region_lst and trace["marker"]['symbol'] == 'circle':
            trace["showlegend"] = True
            region_lst.append(trace["name"])
        else:
            trace["showlegend"] = False
        
    fig.update_traces(
        hovertemplate="<br>".join([
        "Data: %{customdata[0]}",
        "Method: %{customdata[1]}",
        "Accuracy: %{y}",
        "Consistency: %{x}",
            ]))   
    fig.update_traces(line=dict(width=3),marker = dict(size=10),opacity=0.9)
    
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
        
        
def build_cor(data_sel, method_sel,
                 k_sel, criteria_sel,new_data=None
                 ):

 
    dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.K ==k_sel)
            &(df.criteria==criteria_sel)]

    
    this_palette=dict((i,palette[i]) for i in method_sel)
    this_palette_data=dict((i,palette_data[i]) for i in data_sel)
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
          
        
    
def build_line_raw(data_sel, method_sel,
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
                      facet_col="data",facet_col_wrap=3,facet_row_spacing=0.05,
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

        
    return fig
                
def build_scatter_raw(data_sel, method_sel,
                 k_sel, criteria_sel,new_data=None):

 
    dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.K ==k_sel)
            &(df.criteria==criteria_sel)]
    
    
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
    fig = px.scatter(dff, x="Accuracy", y="Consistency", color='method', 
#                      trendline="ols",
                    opacity=0.5,  facet_col="data",facet_col_wrap=3,
                     #width=1000, height=800,
                color_discrete_map=this_palette,
                symbol='method', symbol_map= this_markers_choice,
                 category_orders={"method":list(this_palette.keys())},
               labels=dict(Consistency=criteria_sel, method="Method"),

                )
   
    fig.update_traces(marker_size=10)
    fig.update_xaxes(matches=None,showticklabels=True)
    
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
def build_heat_raw(data_sel, method_sel,
                 k_sel, criteria_sel,new_data=None):
    
    
    
    cross_ave = cross[(cross.data.isin(data_sel))
                &(cross['method1'].isin(method_sel))
                &(cross['method2'].isin(method_sel))
                &(cross.K ==k_sel)
                &(cross.criteria==criteria_sel)]
    cross_ave=cross_ave.groupby(['data','method1','method2','criteria','K'],as_index=False)['value'].mean()
#     sub = cross_ave[(cross_ave['method1'].isin(method_sel))&(cross_ave['method2'].isin(method_sel))]
#     sub = sub[(sub['K']==k_sel)&(sub['criteria']==criteria_sel)]
    method_acc_sel=['LogisticLASSO','LogisticRidge','SVM','Tree','XGB','RF','Occlusion (MLP)']
    
    dff=df[(df.data.isin(data_sel))
                &(df.method.isin(method_acc_sel))
                &(df.K ==k_sel)
                &(df.criteria==criteria_sel)]
    dff=dff.replace({'Occlusion (MLP)':'MLP'})
    dff = dff.groupby(['method','data']).mean().reset_index()
    subss = {}
    for i,dd in enumerate(data_sel):
        hh = cross_ave[cross_ave.data==dd].pivot("method1", "method2", "value")
        hh = hh.fillna(0)+hh.fillna(0).T
        np.fill_diagonal(hh.values, 1)
        hh=round(hh.reindex(columns=method_sel).reindex(method_sel),3)
        
        subss[dd]=hh

    tt =[[i]  for i in data_sel for _ in range(2)]
    tt = [item for sublist in tt for item in sublist]
    this_palette=dict((i,palette[i]) for i in method_sel)
    
    fig = make_subplots(rows=len(data_sel), cols=2,  column_widths=[0.7, 0.3], horizontal_spacing=0.05,
                    vertical_spacing=0.05,                     
                                     subplot_titles=(tt)                                                                  )

    for i,dd in enumerate(data_sel):
        bar1 = px.imshow(subss[dd],text_auto='.2f', origin='lower',)
        bar2 = px.bar(dff[dff.data ==dd], x='method', y='Accuracy',range_y = [0,1],
                        color_discrete_map =palette,color='method',
                     text_auto='.3' )

        for trace in bar1.data:
            fig.add_trace(trace, i+1, 1)
        for trace in bar2.data:
            trace["width"] = 1
            trace["showlegend"] = False

            fig.add_trace(trace, i+1, 2)

        fig.update_traces(coloraxis='coloraxis1',selector=dict(xaxis='x'))
        fig.update_layout(
                      coloraxis=dict(colorscale='Purp', 
                                     showscale = False),)
        fig.update_xaxes(tickangle=45)
    fig['layout'].update(height=6000, width=800)
    return fig

def build_dot(data_sel, method_sel,
                 k_sel, criteria_sel,new_data=None):

 
    dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.K ==k_sel)
            &(df.criteria==criteria_sel)]

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





