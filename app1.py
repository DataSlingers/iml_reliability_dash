Jupyter Notebook
app1.py
Last Tuesday at 4:21 PM
Python
File
Edit
View
Language
1
### Data
2
import pandas as pd
3
import pickle
4
### Graphing
5
import plotly.graph_objects as go
6
### Dash
7
import dash
8
import dash_core_components as dcc
9
import dash_html_components as html
10
import dash_bootstrap_components as dbc
11
from dash.dependencies import Output, Input
12
## Navbar
13
from nav import Navbar
14
import numpy as np
15
import plotly.express as px
16
from dash import dash_table
17
from plotly.subplots import make_subplots
18
import plotly.graph_objects as go
19
​
20
​
21
nav = Navbar()
22
# header = html.H3(
23
#     'Reliability of Feature Importance'
24
# )
25
​
26
# accs=pd.read_csv('feature_impo_accs.csv')
27
df = pd.read_csv("feature_impo.csv")
28
# accs=accs[accs.model!='LogisticLASSO']
29
# df=df.dropna()
30
cross = pd.read_csv('cross_fi.csv')
31
cross_pred=pd.read_csv('fi_cross_pred.csv')
32
puris=pd.read_csv('feature_impo_pur.csv')
33
​
34
# data_options = df['data'].unique().tolist()
35
# method_options = df['method'].unique().tolist()
36
criteria_options = df['criteria'].unique().tolist()
37
method_category_options = ['Selected','Model Specific','Model Agnostic','All']
38
​
39
k_options =df['K'].unique().tolist()
40
plot_summary_options = {'heatmap':'Consistency heatmap across methods',
41
                        'line':'Consistency line plot within methods',
42
                        'heat2':'Consistency heatmap within methods',
43
                        'bump':'Bump plot of the most consistent methods across data sets',
