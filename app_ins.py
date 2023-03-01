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
# from dash import Dash, dcc
## Navbar
from nav import Navbar
import numpy as np
import plotly.express as px
from dash import dash_table
from PIL import Image



nav = Navbar()
section = dcc.Markdown(
'''
# Instruction  
### Dashboard 
   We have interactive results of five sections: 
   
   * Feature importance (Classification), 
   * Feature importance (Regression), 
   * Clustering, 
   * Dimension reduction (Clustering), 
   * Dimension reduction (Local Neighbors) 
   
    Each section has the same layout, with options on the left and summary and raw plots on the right. 
''',
    mathjax=True, dangerously_allow_html=True, style={'marginLeft': '5%', 'width': '90%'})

option = dcc.Markdown(
'''
### Options 

* Users can select Summary Graphs and/or Raw plots of interest. Click 'Submit' bottum to show the figures, click 'Reset' button to unselect all figures. 
''',
    mathjax=True, dangerously_allow_html=True, style={'marginLeft': '5%', 'width': '90%'})

up = dcc.Markdown(
'''
* Users can upload their own data set, which has to satisfy the format requirement. ...
''',
    mathjax=True, dangerously_allow_html=True, style={'marginLeft': '5%', 'width': '90%'})



data = dcc.Markdown(
'''
* Data selection: we have results of all the data sets we used in this empirical study. User can show the results of selected data. 
''',
    mathjax=True, dangerously_allow_html=True, style={'marginLeft': '5%', 'width': '90%'})

meth =dcc.Markdown(
'''
* Method selection: show the results of selected methods. 
''',
    mathjax=True, dangerously_allow_html=True, style={'marginLeft': '5%', 'width': '90%'})

criteria =dcc.Markdown(
'''
* Select criteria: figures are generated with the selected criteria.   
    * Two options in Feature Importance section: RBO and ARI. 
    * Four options in Clustering Importance section: ARI, Mutual Information (MI), fowlkes mallows, and V measuer(v_measure)
    * Four options in Dimension Reduction+Clustering section: ARI, Mutual Information (MI), fowlkes mallows, and V measuer(v_measure)
    * One options in Dimension Reduction+Local Neighbor section: Jaccard similarity. 

''',
    mathjax=True, dangerously_allow_html=True, style={'marginLeft': '5%', 'width': '90%'})


noise=dcc.Markdown(
'''
* Select Noise Level (sigma): control the level of noise addition, where the added noise has variance sigma. 

* Select Noise Type: normal or laplace noise. 
''',
    mathjax=True, dangerously_allow_html=True, style={'marginLeft': '5%', 'width': '90%'})

other=dcc.Markdown(
'''
 * Section Specific options:
     * Select Top K in Feature Importance section: select the number of most important features we want to compare. 
     * Select Rank in Dimension Reduction: select the rank of reduced dimension we aim to evaluate with. 
''',
    mathjax=True, dangerously_allow_html=True, style={'marginLeft': '5%', 'width': '90%'})

### Figures 
heat = dcc.Markdown(
'''
#### Summary figures: 
* Consistency heatmap: the cross-method average consistency of interpretations obtain from each pair of IML methods, averaged over 100 repeats and different data sets.
''',
    mathjax=True, dangerously_allow_html=True, style={'marginLeft': '5%', 'width': '90%'})
line = dcc.Markdown(
'''
* Consistency line plot: the average pairwise consistency of interpretations of an IML method aggregated over 100 repeats, with colors representing different methods. The x-axis is the data sets we used, ordered by # feature/# observation ratio, and the y-axis is the consistency score of this task, ranging in \[0,1\].
''',
    mathjax=True, dangerously_allow_html=True, style={'marginLeft': '5%', 'width': '90%'})
bump = dcc.Markdown(
'''
* Consistency bump plot: ranks IML methods by their consistency in each data set. The methods of the y-axis on the right is ordered by the average consistency across all data sets.
''',
    mathjax=True, dangerously_allow_html=True, style={'marginLeft': '5%', 'width': '90%'})
# fit = dcc.Markdown(
# '''
# * Fit: to evaluate the relationship between prediction accuracy and interpretation consistency. The scatterplot shows the consistency score vs. predictive accuracy, with colors representing different IML methods. The points with the same color represent data sets, averaged over 100 repeats. The fitted regression lines between consistency score and predictive accuracy does not necessarily have positive coefficients.

# ''',
#     mathjax=True, dangerously_allow_html=True, style={'marginLeft': '5%', 'width': '90%'})
# cor = dcc.Markdown(
# '''
# * Cor: to evaluate the relationship between prediction accuracy and interpretation consistency. The histogram plots the correlation between consistency score and predictive accuracy for each method, average over different data sets and 100 repeats.

# ''',
#     mathjax=True, dangerously_allow_html=True, style={'marginLeft': '5%', 'width': '90%'})
dot = dcc.Markdown(
'''
* Consistency \& accuracy scatterplots: the consistency scatterplot demonstrates the consistency score against each IML methods, where scatters represents data sets. The size of the scatters demonstrates the prediction accuracy, where larger size indicates higher accuracy. The accuracy scatterplot switches the values of y-axis and scatter size. It plots the model prediction accuracy against each method, for different data sets, and the size of scatters indicate the interpretation consistency.

''',
    mathjax=True, dangerously_allow_html=True, style={'marginLeft': '5%', 'width': '90%'})
heat_raw = dcc.Markdown(
'''
#### Raw Figures: 
* Raw consistency heatmap and accuracy bar plot: the cross-method average consistency of interpretations in each data set, along with prediction accuracy.  
''',
    mathjax=True, dangerously_allow_html=True, style={'marginLeft': '5%', 'width': '90%'})

scatter = dcc.Markdown(
'''
#### Raw Figures: 
* Raw consistency \& accuracy scatterplots: the consistency score against prediction accuracy for each data set, where scatters represent different IML methods.
''',
    mathjax=True, dangerously_allow_html=True, style={'marginLeft': '5%', 'width': '90%'})



line_raw = dcc.Markdown(
'''
* Raw consistency line plot: the interpretation consistency against some related parameter values of each data set.
    * Featue Importance Section: Line plot of interpretation consistency scores of each data versus number of importance featuers, colored by IML methods.
    
    * Clustering & Dimension Reduction (Clustering) sections: Line plot of interpretation consistency scores of each data versus noise level, colored by IML methods.
    
    * Dimension Reduction (Local Neighbors): Line plot of interpretation consistency scores of each data versus number of neighbors, colored by IML methods.
''',
    mathjax=True, style={'marginLeft': '5%', 'width': '80%'},
)


# Add images
import base64
bar = 'fig/bar.png' # replace with your own image
bar_image = base64.b64encode(open(bar, 'rb').read())

sel = 'fig/sel.png' # replace with your own image
sel_image = base64.b64encode(open(sel, 'rb').read())

upp = 'fig/up.png' # replace with your own image
up_image = base64.b64encode(open(upp, 'rb').read())

dat = 'fig/dat.png' # replace with your own image
data_image = base64.b64encode(open(dat, 'rb').read())

methh = 'fig/meth.png' # replace with your own image
meth_image = base64.b64encode(open(methh, 'rb').read())
crii = 'fig/cri.png' # replace with your own image
cri_image = base64.b64encode(open(crii, 'rb').read())
noisee = 'fig/noise.png' # replace with your own image
noise_image = base64.b64encode(open(noisee, 'rb').read())

heat_sum = 'fig/class_heat.png' # replace with your own image
heat_sum_image = base64.b64encode(open(heat_sum, 'rb').read())

heat_raww = 'fig/heat_raw.png' # replace with your own image
heat_raw_image = base64.b64encode(open(heat_raww, 'rb').read())

line_sum = 'fig/class_line.png' # replace with your own image
line_sum_image = base64.b64encode(open(line_sum, 'rb').read())

line_raww = 'fig/line_raw.png' # replace with your own image
line_raw_image = base64.b64encode(open(line_raww, 'rb').read())

scatterr = 'fig/scatter_raw.png' # replace with your own image
scatter_image = base64.b64encode(open(scatterr, 'rb').read())

dot_sum1 = 'fig/class_fit1.png' # replace with your own image
dot_sum1_image = base64.b64encode(open(dot_sum1, 'rb').read())

dot_sum2 = 'fig/class_fit2.png' # replace with your own image
dot_sum2_image = base64.b64encode(open(dot_sum2, 'rb').read())

bump_sum = 'fig/class_bump.png' # replace with your own image
bump_sum_image = base64.b64encode(open(bump_sum, 'rb').read())

aucc = 'fig/auc.png' # replace with your own image
aucc_image = base64.b64encode(open(aucc, 'rb').read())


auc_raw = 'fig/auc_raw.png' # replace with your own image
auc_raw_image = base64.b64encode(open(auc_raw, 'rb').read())




def App_ins():
    layout = html.Div([
        nav,
        section,
        html.Img(src='data:image/png;base64,{}'.format(bar_image.decode()),style={'text-align': 'center','width': '500px',"margin-left": "80px"}),
        option,
        html.Img(src='data:image/png;base64,{}'.format(sel_image.decode()),style={'text-align': 'center','width': '200px',"margin-left": "80px"}),

       
        up,
        html.Img(src='data:image/png;base64,{}'.format(up_image.decode()),style={'text-align': 'center','width': '200px',"margin-left": "80px"}),
    
       data,
        html.Img(src='data:image/png;base64,{}'.format(data_image.decode()),style={'text-align': 'center','width': '200px',"margin-left": "80px"}),        
        meth,
        html.Img(src='data:image/png;base64,{}'.format(meth_image.decode()),style={'text-align': 'center','width': '200px',"margin-left": "80px"}),
        criteria,
        html.Img(src='data:image/png;base64,{}'.format(cri_image.decode()),style={'text-align': 'center','width': '200px',"margin-left": "80px"}),
        noise,
        html.Img(src='data:image/png;base64,{}'.format(noise_image.decode()),style={'text-align': 'center','width': '200px',"margin-left": "80px"}),
        other,
        
        ## summary figures

        line,
        html.Img(src='data:image/png;base64,{}'.format(line_sum_image.decode()),style={'text-align': 'center','width': '500px',"margin-left": "200px"}),
        bump,
        html.Img(src='data:image/png;base64,{}'.format(bump_sum_image.decode()),style={'text-align': 'center','width': '500px',"margin-left": "200px"}),
        heat,
        html.Img(src='data:image/png;base64,{}'.format(heat_sum_image.decode()),style={'text-align': 'center','width': '500px',"margin-left": "200px"}),
        dot,
        html.Img(src='data:image/png;base64,{}'.format(dot_sum1_image.decode()),style={'text-align': 'center','width': '500px',"margin-left": "200px"}),
        html.Img(src='data:image/png;base64,{}'.format(dot_sum2_image.decode()),style={'text-align': 'center','width': '500px',"margin-left": "200px"}),
        
        ## raw figures
        heat_raw, 
        html.Img(src='data:image/png;base64,{}'.format(heat_raw_image.decode()),style={'text-align': 'center','width': '500px',"margin-left": "200px"}),
        line_raw, 
        html.Img(src='data:image/png;base64,{}'.format(line_raw_image.decode()),style={'text-align': 'center','width': '500px',"margin-left": "200px"}),
        scatter,
            html.Img(src='data:image/png;base64,{}'.format(scatter_image.decode()),style={'text-align': 'center','width': '500px',"margin-left": "200px"}),
    
    
    
    
    ])
    
    return layout















