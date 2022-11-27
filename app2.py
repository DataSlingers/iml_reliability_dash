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

df = pd.read_csv("clustering.csv")
cross = pd.read_csv('cross_clus.csv')

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
            ###############################
            
            html.P("Select: Noise Type"),
            dcc.RadioItems(
                id="noise-select_clus",
                options=[{"label": i, "value": i} for i in noise_options],
                value=noise_options[1],
            ),
                
            html.Hr(),            
            html.P("Select: Noise Level (sigma)"),
            dcc.Dropdown(
                id="sigma-select_clus",
                options=[{"label": i, "value": i} for i in sigma_options],
                value=1,
            
            ),            
            
            html.Hr(),

            html.P("Select: Consistency Metric"),
            dcc.RadioItems(
                id="criteria-select_clus",
                options=[{"label": i, "value": i} for i in criteria_options],
                value=criteria_options[0],
            ),        
                
            html.Hr(),

            html.P("Select: Interpretability Method"),
            dcc.Dropdown(
                id="method-select_clus",
                options=[{"label": i, "value": i} for i in meths],
                value=meths[0:10],
                multi=True,
            ),
            html.Br(),
            
            
            
            
            html.Hr(),
            html.P("Select: Data Sets"),
            dcc.Dropdown(
                id="data-select_clus",
                options=[{"label": i, "value": i} for i in data_options],
                value=data_options[:6],
                multi=True,
            ),
            html.Br(),
            html.Hr(),

   
    

            #################################
            ########### select figures 
            #################################

            html.Hr(),
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

            

            html.Hr(),
           
            dbc.Button('Submit', id='submit-button',n_clicks=0, color="primary",className="me-1"),
            dbc.Button('Reset',id='reset-button',n_clicks=0, color="secondary",className="me-1"),
            html.Hr(),
   
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
            html.Div(id='title_summary'),
            html.Div(id='subtitle_summary'),
            html.Div(id='show_heatmap'),
            html.Div(id='show_line'),
            html.Div(id='show_bump'),
#             html.Div(id='show_fit'),
            html.Div(id='show_dot'),
            html.Div(id='show_cor'),
            ######### raw plots 
            html.Div(id='title_summary_raw'),
            html.Div(id='show_line_raw'),
            html.Div(id='show_scatter_raw'),
            html.Div(id='show_heatmap_raw'),

           
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
        


def build_heat_summary_clus(data_sel,method_sel,
                 criteria_sel,noise_sel,sigma_sel
                 ):

        cross_ave = cross[cross.data.isin(data_sel)]
        cross_ave=cross_ave.groupby(['method1','method2','criteria','noise','sigma'],as_index=False)['value'].mean()
        sub = cross_ave[(cross_ave['method1'].isin(method_sel))&(cross_ave['method2'].isin(method_sel))]
        sub = sub[(sub['sigma']==sigma_sel)&(sub['criteria']==criteria_sel)&(sub['noise']==noise_sel)]
        sub = sub.pivot("method1", "method2", "value")
        sub = sub.fillna(0)+sub.fillna(0).T
        np.fill_diagonal(sub.values, 1)
        sub=round(sub.reindex(columns=method_sel).reindex(method_sel),3)
        
        
        fig = px.imshow(sub, text_auto=True, aspect="auto",color_continuous_scale='Purp', origin='lower',
               labels=dict(x="Method", y="Method", color="Consistency"))
        fig.update_xaxes(tickangle=45)
        fig.layout.coloraxis.showscale = False
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

    this_palette=dict((i,palette[i]) for i in method_sel)
    this_line_choice=dict((i,line_choice[i]) for i in method_sel)
    this_palette_data =  [i for i in palette_data.keys() if i in data_sel]   

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
    this_palette=dict((i,palette[i]) for i in method_sel)
    this_markers_choice=dict((i,markers_choice[i]) for i in method_sel)

    dff=df[(df.data.isin(data_sel))
                &(df.method.isin(method_sel))
                &(df.noise ==noise_sel)
                &(df.sigma ==float(sigma_sel))
                &(df.criteria==criteria_sel)]
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

def build_bump_clus(data_sel, method_sel,
                 criteria_sel,noise_sel,sigma_sel,new_data=None
                 ):
####### filter data
    dff=df[(df.data.isin(data_sel))
                &(df.method.isin(method_sel))
                &(df.noise ==noise_sel)
                &(df.sigma ==float(sigma_sel))
                &(df.criteria==criteria_sel)]
    this_palette=dict((i,palette[i]) for i in method_sel)
    this_markers_choice=dict((i,markers_choice[i]) for i in method_sel)

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
               labels=dict(Consistency=criteria_sel, method="Method"),
                                     

                custom_data=['data','method'],
                )
    fig.update_traces(
        hovertemplate="<br>".join([
        "Data: %{customdata[0]}",
        "Method: %{customdata[1]}",
        "Accuracy: %{x}",
        "Consistency: %{y}",
            ]))
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
    return fig                     
                     

def build_acc_bar_clus(data_sel, method_sel,criteria_sel,noise_sel,sigma_sel):
    this_palette=dict((i,palette[i]) for i in method_sel)
    dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.noise ==noise_sel)
            &(df.sigma ==float(sigma_sel))
            &(df.criteria==criteria_sel)] 

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

def build_line_raw_clus(data_sel, method_sel,
                 criteria_sel, noise_sel,sigma_sel,new_data=None):

    dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.noise ==noise_sel)
            &(df.criteria==criteria_sel)] 
       


    this_palette=dict((i,palette[i]) for i in method_sel)
    this_line_choice=dict((i,line_choice[i]) for i in method_sel)
    this_palette_data =  [i for i in palette_data.keys() if i in data_sel]   

    ###### input new data
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        neww = new_data[
            (new_data.noise ==noise_sel)&
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
                      facet_col="data",facet_col_wrap=3,facet_row_spacing=0.15,
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
                
def build_scatter_raw_clus(data_sel, method_sel,
                 criteria_sel, noise_sel,sigma_sel,new_data=None
                 ):

    dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.noise ==noise_sel)
            &(df.sigma ==float(sigma_sel))
            &(df.criteria==criteria_sel)] 
  
    
    this_palette=dict((i,palette[i]) for i in method_sel)
    this_markers_choice=dict((i,markers_choice[i]) for i in method_sel)
    this_palette_data =  [i for i in palette_data.keys() if i in data_sel]   

    ###### input new data
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        neww = new_data[
            (new_data.noise ==noise_sel)&
            (new_data.sigma ==float(sigma_sel))
            (new_data.criteria==criteria_sel)]
        dff = pd.concat([dff, neww]) 
        for mm in set(new_data['method']):
            this_palette[mm]='black'
            this_markers_choice[mm]='star'
            
            
            
    fig = px.scatter(dff, x="Accuracy", y="Consistency", color='method', 
#                      trendline="ols",
                     facet_col="data",facet_col_wrap=3,
                     #width=1000, height=800,
                color_discrete_map=this_palette,
                symbol='method', symbol_map= this_markers_choice,
                 category_orders={"method":list(this_palette.keys())},
               labels=dict(Consistency=criteria_sel, method="Method"),

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
           
def build_heat_raw_clus(data_sel, method_sel,
                 criteria_sel, noise_sel,sigma_sel,new_data=None
                 ):
    
    
    
    cross_ave = cross[(cross.data.isin(data_sel))
                &(cross['method1'].isin(method_sel))
                &(cross['method2'].isin(method_sel))
                &(cross.noise==noise_sel)
                &(cross.sigma==sigma_sel)
                &(cross.criteria==criteria_sel)]
    cross_ave=cross_ave.groupby(['data','method1','method2'],as_index=False)['value'].mean()
#     sub = cross_ave[(cross_ave['method1'].isin(method_sel))&(cross_ave['method2'].isin(method_sel))]
#     sub = sub[(sub['K']==k_sel)&(sub['criteria']==criteria_sel)]

    dff=df[(df.data.isin(data_sel))
                &(df.method.isin(method_sel))
                &(df.noise==noise_sel)
                &(df.sigma==sigma_sel)
                &(df.criteria==criteria_sel)]
    
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

    fig = make_subplots(rows=9, cols=2, horizontal_spacing=0.05,
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
    fig['layout'].update(height=4000, width=800)
    return fig

def build_dot_clus(data_sel, method_sel,
                 criteria_sel, noise_sel,sigma_sel,new_data=None
                 ):

    dff=df[(df.data.isin(data_sel))
            &(df.method.isin(method_sel))
            &(df.noise ==noise_sel)
            &(df.sigma ==float(sigma_sel))
            &(df.criteria==criteria_sel)] 
    dff['size1']=(dff['Accuracy']**2)
    dff['size1']=[max(i,0.1) for i in dff['size1']]
    dff['size2']=(dff['Consistency']**2)
    dff['size2']=[max(i,0.1) for i in dff['size2']]
    ###### input new data
    if new_data is not None:
        new_data = pd.DataFrame(new_data)
        neww = new_data[
            (new_data.noise ==noise_sel)&
            (new_data.sigma ==float(sigma_sel))&
            (new_data.criteria==criteria_sel)]
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





