from plotly.subplots import make_subplots


import seaborn as sns
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




def display_figure(pp,plot_selected,pathname):
#     if click is None:
#         raise PreventUpdate
 
#     if click>0:
    paths = {'/clustering':'clus',
             '/dimension_reduction_clustering':'dr',
               '/knn':'knn'
            }
    #     heads = {'heatmap':'Interpretations are unreliable (cross methods)',
    #              'line':'Interpretations are unreliable (within methods)',
    #              'bump':'Each data has its own most consistent method (No free lunch)',
    #              'fit':'Predictive accuracy does not lead to consistent interpretation',
    #              'cor':'Predictive accuracy does not lead to consistent interpretation',
    #              'line_new':'Line with new data',
    #              'bump_new':'Bump with new data',
    #              'fit_new':'Fit with new data',
    #              'cor_new':'Cor with new data',

    #              'line_raw':'Line plot of Raw Results',
    #              'scatter_raw':'Scatter plot of Raw Results',
    #              'k_raw': 'Consistency vs. number of local neighbors',
    #              'heatmap_raw': 'cross method consistency & accuracy'
    #             }
    heads = {'heatmap':'Summary Figure for Q2: Consistency Heatmap (Between methods, aggregated over data sets)',
                 'line':'Additional Summary Figure for Q1: Consistency Lineplot (Within methods)',
                 'heat2':'Summary Figure for Q1: Consistency Heatmap (Within methods)',
                 # (across data sets)',
                 'bump':'Summary Figure for Q1: Consistency Bump Plot (Within methods)',
                 #of the most consistent methods across data sets',
                'fit':'Summary Figure for Q3: Consistency & Accuracy Scatterplot (All data sets)', 
                 #vs. predictive accuracy',
#                  'dot':'Summary Figure: Consistency & Accuracy Scatterplot (aggregated over data sets)', 
#                  'cor':'Summary Figure: Correlation between onsistency and predictive accuracy',
                 
                 ##new data
                 'line_new':'Consistency Lineplot plot with new data (Within methods)',
                 'heat2_new':'Consistency Heatmap with new data (Within methods)',
                 'bump_new':'Consistency Bump Plot with new data (Within methods) ',
#                  'fit_new':'Consistency & Accuracy Scatterplot with new data (All data sets)',

                 'line_raw':'Detailed Figure for Q1: Consistency Lineplot wrt. top K features (all data sets)',
                 #vs number of features for all data sets',
                 'line_raw2':'Detailed Figure for Q1: Consistency Lineplot wrt. noise level (all data sets)',
                 # vs. noise level for all data sets',
                 'scatter_raw':'Detailed Figure for Q3: Consistency Scatterplot wrt. predictive accuracy (all data sets)',
                 'k_raw': 'Detailed Figure for Q1: Consistency Lineplot wrt. number of local neighbors (all data sets)',
                 'heatmap_raw': 'Detailed Figure for Q2: Consistency Heatmap across methods (Between methods, all data sets)',
#                  'acc_raw': 'Raw Results: Prediction Accuracy Boxplot'
                }


    describ = {'heatmap':['Among different methods, we aim to evaluate whether different methods would result in similar interpretations, the heatmap shows the cross-method average consistency of interpretations obtain from each pair of IML methods. For example, the cell of method i and method j represents the consistency between the interpretations of i and j, averaged over 100 repeats and different data sets.'],
                 'line':['Within each method, we aim to measure whether interpretations are consistent among repeats. The line plot shows the data sets versus the average pairwise consistency of 100 repeats of an IML method, with colors representing different methods. The x-axis is the data sets we used, ordered by # feature/# observation ratio, and the y-axis is the consistency score of this task, ranging in [0,1]. '],
                   'heat2':['Within each method, we aim to measure whether interpretations are consistent among repeats. The heatmap shows the data sets versus the average pairwise consistency of 100 repeats of an IML method. The x-axis is the data sets we used, ordered by # observation/# feature ratio, and the y-axis is the consistency score of this task.'],
                   
                 'bump':['The bump plot ranks IML methods by their consistency score for each data, averaged over 100 repeats.'],
                     'fit':['The scatterplot shows the consistency score vs. predictive accuracy, with colors representing different IML methods. The points with the same color represent data sets, averaged over 100 repeats. The fitted regression lines between consistency score and predictive accuracy does not necessarily have positive coefficients.'],
                   'dot':['The scatterplot of consistency/accuracy vs. methods, colored by data and sized by accuracy/consistency.'],

                 'cor':['The histogram plots the correlation between consistency score and predictive accuracy for each method, average over different data sets and 100 repeats. '],
                 'line_new':['Consistency line plot with new data'],
                 'heat2_new':['Consistency heatmap with new data'],
                 'bump_new':['Consistency bump [plot with new data'],
                 'fit_new':['Consistency & Accuracy Scatterplot with new data (All data sets)'],
                 'line_raw':['Line plot of interpretation consistency scores of each data, colored by IML methods. '],
                 'line_raw2':['Line plot of interpretation consistency scores of each data, colored by IML methods. '],
                 'scatter_raw':['Scatter plots of interpretation consistency scores vs. predictive accuracy for each data set, colored by IML methods. '],
                   'k_raw':['Line plots of interpretation consistency scores vs. number of local neighbors K for each data set, colored by IML methods. '],
                 'heatmap_raw':['Consistency heatmap cross methods for each data. '],
                 'acc_raw': ['Prediction Accuracy Boxplot'],
                    }


        
    if pp in plot_selected:
            
        if pp=='fit': 
            fig_id1 = 'fit1_'+paths[pathname] if pathname in paths else 'fit1'
            fig_id2 = 'fit2_'+paths[pathname] if pathname in paths else 'fit2'
            fig_id3 = 'pv1_'+paths[pathname] if pathname in paths else 'pv1'
            fig_id4 = 'pv2_'+paths[pathname] if pathname in paths else 'pv2'
            if pathname !='/clustering'and pathname !='/dimension_reduction_clustering' :
#                 if fig_id1=='fit1' or fig_id1=='fit1_reg':
                return html.Div([
                            html.Details([
                            html.Summary(heads[pp],style={'color':'midnightblue','fontSize':'25px'}),
                            html.Div([
                                   html.Details([
                                    html.Summary('Description'),
                                    html.Div(children=describ[pp], className='desc',
                                             id='my-description')
                                ],
                                    id="desc-dropdown",
                                    open=False
                                 ),

                            dls.Hash(                        
                            dcc.Graph(id=fig_id1,
                                  style={'width': '180vh', 'height': '60vh'}
                                     ),
                            color="#435278",
                            speed_multiplier=2,
                            size=100,
                                ),


                            dls.Hash(                        
                            dcc.Graph(id=fig_id3,
                                  style={'width': '150vh', 'height': '70vh'}
                                     ),
                            color="#435278",
                            speed_multiplier=2,
                            size=100,
                                ),      


                            dls.Hash(                        
                            dcc.Graph(id=fig_id2,
                                  style={'width': '190vh', 'height': '60vh'}
                                     ),
                            color="#435278",
                            speed_multiplier=2,
                            size=100,
                                ),


                            dls.Hash(                        
                            dcc.Graph(id=fig_id4,
                                  style={'width': '150vh', 'height': '70vh'}
                                     ),
                            color="#435278",
                            speed_multiplier=2,
                            size=100,
                                ),      


                            ])
                        ],
                            id="desc-dropdown",
                            open=True
                        ), 
                ])


            else:
                return html.Div([
                            html.Details([
                            html.Summary(heads[pp],style={'color':'midnightblue','fontSize':'25px'}),
                            html.Div([
                                   html.Details([
                                    html.Summary('Description'),
                                    html.Div(children=describ[pp], className='desc',
                                             id='my-description')
                                ],
                                    id="desc-dropdown",
                                    open=False
                                 ),

                            dls.Hash(                        
                            dcc.Graph(id=fig_id1,
                                  style={'width': '80vh', 'height': '60vh'}
                                     ),
                            color="#435278",
                            speed_multiplier=2,
                            size=100,
                                ),


                            dls.Hash(                        
                            dcc.Graph(id=fig_id3,
                                  style={'width': '80vh', 'height': '70vh'}
                                     ),
                            color="#435278",
                            speed_multiplier=2,
                            size=100,
                                ),      


                            dls.Hash(                        
                            dcc.Graph(id=fig_id2,
                                  style={'width': '80vh', 'height': '60vh'}
                                     ),
                            color="#435278",
                            speed_multiplier=2,
                            size=100,
                                ),


                            dls.Hash(                        
                            dcc.Graph(id=fig_id4,
                                  style={'width': '80vh', 'height': '70vh'}
                                     ),
                            color="#435278",
                            speed_multiplier=2,
                            size=100,
                                ),      


                            ])
                        ],
                            id="desc-dropdown",
                            open=True
                        ), 
                ])
#                     return html.Div([
#                                 html.Details([
#                                 html.Summary(heads[pp],style={'color':'midnightblue','fontSize':'25px'}),
#                                 html.Div([
#                                        html.Details([
#                                         html.Summary('Description'),
#                                         html.Div(children=describ[pp], className='desc',
#                                                  id='my-description')
#                                     ],
#                                         id="desc-dropdown",
#                                         open=False
#                                      ),

#                                 dls.Hash(                        
#                                 dcc.Graph(id=fig_id1,
#                                       style={'width': '60vh', 'height': '40vh'}
#                                          ),
#                                 color="#435278",
#                                 speed_multiplier=2,
#                                 size=100,
#                                     ),
 
#                                 dls.Hash(                        
#                                 dcc.Graph(id=fig_id2,
#                                       style={'width': '70vh', 'height': '40vh'}
#                                          ),
#                                 color="#435278",
#                                 speed_multiplier=2,
#                                 size=100,
#                                     ),
             
#                                 ])
#                             ],
#                                 id="desc-dropdown",
#                                 open=True
#                             ), 
#                     ])
 
                
        else:
            fig_id = pp+'_'+paths[pathname] if pathname in paths else pp

#             if pp=='dot':
            if pathname not in paths: ## feature importance heatmap needs more space
                if pp=='heatmap':
                    return html.Div([
                            html.Details([
                            html.Summary(heads[pp],style={'color':'midnightblue','fontSize':'25px'}),
                            html.Div([
                                   html.Details([
                                    html.Summary('Description'),
                                    html.Div(children=describ[pp], className='desc',
                                             id='my-description')
                                ],
                                    id="desc-dropdown",
                                    open=False
                                 ),


                            dls.Hash(                        
                            dcc.Graph(id=fig_id,
                                  style={'width': '120vh', 'height': '70vh'}
                                     ),
                            color="#435278",
                            speed_multiplier=2,
                            size=100,
                                ),
                            ])
                        ],
                            id="desc-dropdown",
                            open=True
                        ), 
                ])

            if pp == 'line' or pp=='line_new' or pp=='heat2' or pp=='heat2_new':
                if pathname !='/knn':
                    this_width = '170vh'
                    this_height = '70vh' 
                else:
                    this_width = '100vh'
                    this_height = '70vh' 
                    
            elif pp == 'scatter_raw' or pp=='line_raw' or pp=='k_raw' or pp=='acc_raw':
                if pp=='line_raw' and pathname in paths:
                    pp='line_raw2'
                this_height = '150vh'
                this_width = '180vh'

            elif pp == 'heatmap_raw':
#                     this_height = '500vh'
                #this_width = '80vh'             
                 this_height = '300vh'
                 this_width = '180vh'
            else:
                this_width = '100vh'
                this_height = '70vh'                   

            return html.Div([
                            html.Details([
                            html.Summary(heads[pp],style={'color':'midnightblue','fontSize':'25px'}),
                            html.Div([
                                   html.Details([
                                    html.Summary('Description'),
                                    html.Div(children=describ[pp], className='desc',
                                             id='my-description')
                                ],
                                    id="desc-dropdown",
                                    open=False
                                 ),


                            dls.Hash(                        
                            dcc.Graph(id=fig_id,
                                  style={'width': this_width, 'height': this_height}
                                     ),
                            color="#435278",
                            speed_multiplier=2,
                            size=100,
                                ),
                            ])
                        ],
                            id="desc-dropdown",
                            open=True
                        ), 
                ])                    

           

#                 else: ## raw heatmaps 
                    
#                     return html.Div([
#                                 html.Details([
#                                 html.Summary(heads[pp],style={'color':'midnightblue','fontSize':'25px'}),
#                                 html.Div([
#                                        html.Details([
#                                         html.Summary('Description'),
#                                         html.Div(children=describ[pp], className='desc',
#                                                  id='my-description')
#                                     ],
#                                         id="desc-dropdown",
#                                         open=False
#                                      ),


#                                 dls.Hash(                        
#                                 dcc.Graph(id=fig_id,
#                                       style={'width': '80vh', 'height': '500vh'}
#                                          ),
#                                 color="#435278",
#                                 speed_multiplier=2,
#                                 size=100,
#                                     ),
#                                 ])
#                             ],
#                                 id="desc-dropdown",
#                                 open=True
#                             ), 
#                     ])




# def display_figure(pp,plot_selected, click,pathname):
#     if click is None:
#         raise PreventUpdate
 
#     if click>0:
#         paths = {'/clustering':'clus',
#                  '/dimension_reduction_clustering':'dr',
#                    '/knn':'knn'
#                 }
#         heads = {'heatmap':'Interpretations are unreliable (cross methods)',
#                  'line':'Interpretations are unreliable (within methods)',
#                  'bump':'Each data has its own most consistent method (No free lunch)',
#                  'fit':'Predictive accuracy does not lead to consistent interpretation',
#                  'cor':'Predictive accuracy does not lead to consistent interpretation',
#                  'line_new':'Line with new data',
#                  'bump_new':'Bump with new data',
#                  'fit_new':'Fit with new data',
#                  'cor_new':'Cor with new data',
                 
#                  'line_raw':'Line plot of Raw Results',
#                  'scatter_raw':'Scatter plot of Raw Results',
#                  'k_raw': 'Consistency vs. number of local neighbors'
#                 }
        
        
        
#         describ = {'heatmap':['Among different methods, we aim to evaluate whether different methods would result in similar interpretations, the heatmap shows the cross-method average consistency of interpretations obtain from each pair of IML methods. For example, the cell of method i and method j represents the consistency between the interpretations of i and j, averaged over 100 repeats and different data sets.'],
#                  'line':['Within each method, we aim to measure whether interpretations are consistent among repeats. The line plot shows the data sets versus the average pairwise consistency of 100 repeats of an IML method, with colors representing different methods. The x-axis is the data sets we used, ordered by # feature/# observation ratio, and the y-axis is the consistency score of this task, ranging in [0,1]. '],
#                  'bump':['The bump plot ranks IML methods by their consistency score for each data, averaged over 100 repeats.'],
#                      'fit':['The scatterplot shows the consistency score vs. predictive accuracy, with colors representing different IML methods. The points with the same color represent data sets, averaged over 100 repeats. The fitted regression lines between consistency score and predictive accuracy does not necessarily have positive coefficients.'],
#                  'cor':['The histogram plots the correlation between consistency score and predictive accuracy for each method, average over different data sets and 100 repeats. '],
#                  'line_new':['Line with new data'],
#                  'bump_new':['Bump with new data'],
#                  'fit_new':['Fit with new data'],
#                  'cor_new':['Cor with new data'],
#                  'line_raw':['Line plot of interpretation consistency scores of each data, colored by IML methods. '],
#                  'scatter_raw':['Scatter plots of interpretation consistency scores vs. predictive accuracy for each data set, colored by IML methods. '],
#                    'k_raw':['Line plots of interpretation consistency scores vs. number of local neighbors K for each data set, colored by IML methods. ']
            
            
            
#                     }
        
        
#         if pp in plot_selected and pp=='heatmap':
        
#             fig_id = pp+'_'+paths[pathname] if pathname in paths else pp
# #             if pp=='heatmap':
#             id2 = 'acc_'+paths[pathname] if pathname in paths else 'acc'

#             return html.Div([
#                     html.B(heads[pp]),
#                     html.Details([
#                         html.Summary('Description'),
#                         html.Div(children=describ[pp], className='desc',
#                                  id='my-description')
#                     ],
#                         id="desc-dropdown",
#                         open=False
#                     ), 

#                 dbc.Col([
#                     html.Div(
#                         dls.Hash(                        
#                             dcc.Graph(id=fig_id,
#                                   style={'width': '80vh', 'height': '50vh'}),
#                             color="#435278",
#                             speed_multiplier=2,
#                             size=100,
#                         )
#                     )],width={"size": 4},),

#                 dbc.Col([
#                      html.Div(
#                     dls.Hash(                        
#                         dcc.Graph(id=id2,
#                               style={'width': '80vh', 'height': '30vh'}),
#                         color="#435278",
#                         speed_multiplier=2,
#                         size=50,

#                     )
#                    )],width={"size": 4},),


#                     html.Hr(),  # horizontal line

#                 ])

            
#         if pp in plot_selected and pp!='heatmap':
#             fig_id = pp+'_'+paths[pathname] if pathname in paths else pp
#             return html.Div([
#                     html.B(heads[pp]),
#                     html.Details([
#                         html.Summary('Description'),
#                         html.Div(children=describ[pp], className='desc',
#                                  id='my-description')
#                     ],
#                         id="desc-dropdown",
#                         open=False
#                     ), 
#                     dls.Hash(                        
#                         dcc.Graph(id=fig_id,
#                               style={'width': '80vh', 'height': '50vh'}),
#                         color="#435278",
#                         speed_multiplier=2,
#                         size=100,
#                     ),

#                     html.Hr(),  # horizontal line
#                 ])

                
                
# #                 return html.Div([
# #                         html.B(heads[pp]),
# #                         html.Details([
# #                             html.Summary('Description'),
# #                             html.Div(children=describ[pp], className='desc',
# #                                      id='my-description')
# #                         ],
# #                             id="desc-dropdown",
# #                             open=False
# #                         ), 
                    
                    
                    
# #                         dls.Hash(                        
# #                             dcc.Graph(id=fig_id,
# #                                   style={'width': '80vh', 'height': '50vh'}),
# #                             color="#435278",
# #                             speed_multiplier=2,
# #                             size=100,
# #                         ),
# #                         dls.Hash(                        
# #                             dcc.Graph(id=id2,
# #                                   style={'width': '80vh', 'height': '30vh'}),
# #                             color="#435278",
# #                             speed_multiplier=2,
# #                             size=100,
# #                         ),
# #                         html.Hr(),  # horizontal line
# #                     ])
        
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
        html.Div(id='title_summary_new'),
        html.Div(id='show_line_new'),
        html.Div(id='show_heat2_new'),
        html.Div(id='show_bump_new'),
#         html.Div(id='C'),
#         html.Div(id='show_cor_new'),

   
#         html.B('Line with new data'),
#         dcc.Graph(id="line_new_"+paths[pathname] if pathname in paths else "line_new",
#                               style={'width': '80vh', 'height': '50vh'}),
#         html.Hr(),  # horizontal line
#         html.B('Bump with new data'),
#         dcc.Graph(id="bump_new_"+pathfs[pathname] if pathname in paths else "bump_new",
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

        
        