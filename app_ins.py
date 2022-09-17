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



nav = Navbar()
body = dcc.Markdown(
'''
# Instruction  
### Dash Board 
   We have interactive results of three sections: 
   
   * feature importance (classification and regression), 
   * clustering, 
   * dimension reduction (+clustering, +local neighbors) 
   
    Each section has the same layout, with options on the left and summary and raw plots on the right. 
       
### Options 

* Users can select Summary Graphs and/or Raw plots of interest. Click 'Submit' bottum to show the figures, click 'Reset' button to unselect all figures. 

* Users can upload their own data set, which has to satisfy the format requirement. ...

* Data selection: we have results of all the data sets we used in this empirical study. User can show the results of selected data. 

* Method selection: show the results of selected methods. 

* Select criteria: figures are generated with the selected criteria.   
    * There are two options in Feature Importance section: RBO and ARI. 
    * There are four options in Clustering Importance section: ARI, Mutual Information (MI), fowlkes mallows, and V measuer(v_measure)
    * There are four options in Dimension Reduction+Clustering section: ARI, Mutual Information (MI), fowlkes mallows, and V measuer(v_measure)
    *  There is one options in Dimension Reduction+Local Neighbor section: Jaccard similarity. 


* Select Noise Level (sigma): control the level of noise addition, where the added noise has variance sigma. 

* Select Noise Type: normal or laplace noise. 

* Section Specific options:
 * Select Top K in Feature Importance section: select the number of most important features we want to compare. 
 * Select Rank in Dimension Reduction: select the rank of reduced dimension we aim to evaluate with. 


### Figures 

#### Summary figures: 
* Heatmap: to evaluate the consistency of interpretations among different methods. The heatmap shows the cross-method average consistency of interpretations obtain from each pair of IML methods. For example, the cell of method i and method j represents the consistency between the interpretations of i and j, averaged over 100 repeats and different data sets.

* Line: to evaluate consisteny of interpretations of one method among repeats. The line plot shows the data sets versus the average pairwise consistency of 100 repeats of an IML method, with colors representing different methods. The x-axis is the data sets we used, ordered by # feature/# observation ratio, and the y-axis is the consistency score of this task, ranging in \[0,1\].

* Bump: to evaluate the consistency of methods among different data sets. The bump plot ranks IML methods by their consistency score for each data, averaged over 100 repeats. 

* Fit: to evaluate the relationship between prediction accuracy and interpretation consistency. The scatterplot shows the consistency score vs. predictive accuracy, with colors representing different IML methods. The points with the same color represent data sets, averaged over 100 repeats. The fitted regression lines between consistency score and predictive accuracy does not necessarily have positive coefficients.

* Cor: to evaluate the relationship between prediction accuracy and interpretation consistency. The histogram plots the correlation between consistency score and predictive accuracy for each method, average over different data sets and 100 repeats.


#### Raw Figures: 
* Scatter_raw: to evaluate the relationship between prediction accuracy and interpretation consistency for each data set. The scatterplots show interpretation consistency scores vs. predictive accuracy for each data set, colored by IML methods.

* Line_raw: to evaluate the change of consistency under different settings. 
    * Featue Importance Section: Line plot of interpretation consistency scores of each data versus number of importance featuers, colored by IML methods.
    
    * Clustering & Dimension Reduction+Clustering sections: Line plot of interpretation consistency scores of each data versus noise level, colored by IML methods.
    
    * Dimension Reduction+Local Neighbors: Line plot of interpretation consistency scores of each data versus number of neighbors, colored by IML methods.




''',
    mathjax=True, style={'marginLeft': '5%', 'width': '80%'},
)



def App_ins():
    layout = html.Div([
        nav,
        body,
    ])
       
    return layout















