##### import packages 
import dash
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



from nav import *
#from ipynb.fs.full.navbar import Navbar
nav = Navbar()
body = dbc.Container([
#        html.H1('Can We Trust Machine Learning Interpretations? A Reliability Study'),
       html.H3("Welcome to the Interpretable Machine Learning Dashboard"),

       dbc.Row(
           [
               dbc.Col(
                  [
                    
                     html.P(
                         """\
                        Explore IML reliability. """

                           ),
                           dbc.Button("Go to Package Github", color="secondary", href='https://github.com/DataSlingers/iml_reliability_dash'),
                   ],
                  md=4,
               ),
#               dbc.Col(
#                  [
#                      html.H2("some pretty figure"),
#                      dcc.Graph(
#                          figure={"data": [{"x": [1, 2, 3], "y": [1, 4, 9]}]}
#                             ),
#                         ]
#                      ),
                 ]
            )
       ],
className="mt-4",style={'marginLeft': '10%', 'width': '90%'}
)
content= dcc.Markdown(
 
    
'''
### Goal
   **We aim to assess the reliability of interpretabilities of interpretable machine learning (IML) models through a clear large-scale empirical study. **


### This Dashboard

This dashboard provides an interactive platform to present our results. In addition, we provide functions for users to evaluate their own data set and/or IML methods of interests. 

### Study Design 

 A reliable machine learning model should not be over over-sensitive to small changes in the data or parameters of the model. To this end, we evaluate the interpretations' reliability of IML methods by leveraging randomized reliability tests:

* 1. generate multiple randomly perturbed data sets. Details of perturbation techniques; 
 
 * 2. apply each IML method of interest to each perturbed data, and record the generated interpretations as well as the predictive accuracy;

 * 3. measure the reliability of interpretation by the consistency of an IML's interpretations across different perturbed data (within-method consistency), and the consistency among different methods' interpretations on each perturbed data (cross-method consistency). 
 
 Specifically, we ask two questions about the consistency of interpretations: 
 
 * (1) Within-method consistency: if we use different perturbed data, can one IML method yields similar interpretations? and

 * (2) Across-methods consistency: do two IML methods generate similar interpretations on the same data?
 
 We also explore the relationship between interpretation consistency and prediction accuracy to address the third question: 
 
 * (3) Does higher accuracy lead to more consistent interpretations? 



### IML Tasks
We focus on three major types machine learning tasks, including: 

* Feature Importance 

* Clustering (interpretability of observations groups)

* Dimension Reduction (provide interpretations of observations patterns)


### Reliability  Tests

   We define the interpretability to be reliable if the same or similar interpretations can be derived from new data of the same distribution. Therefore, the reliability of a machine learning model can be measured by the consistency of its derived interpretations, and a reliable machine learning model should not be over over-sensitive to small changes in the data or parameters of the model. To this end, the first step of our framework is to design sensitivity tests to obtain interpretations from machine learning models under different circumstance. 

   We have two types of sensitivity test:

   * Random train/test splits

   * Noise addition. 

   The random train/test splits is used for all three tasks. For example, the predictive model has reliable interpretability if the resulting feature importance scores can remain unchanged and consistent with different random train/test splits. Moreover, in unsupervised learning, the consistency of interpretations and its resistance to additional noise can be obtained by adding random noise to the data of interest. With increasing levels of noise added, we are able to illustrate how the consistency would change with more difficult data. Additionally, we may also test how the interpretations changes with different parameter settings.  

### Interpretation Consistency Metrics
We include three categories of interpretabilities: 1). feature importance/ranking derived from supervised learning; 2). clustering results, which implies underlying connections among observations; 3). interpretations in reduced dimension. In each category, we implement robust metrics to measure the reliability of the resulting interpretations. 

#### Feature Importance Metrics
As researcher are generally more interested in the most important features, we focus on top-K rank consistency in the case of feature importance. We consider three metrics: Jaccard similarity [[1]](#1), Kendall's Tau distance [[2]](#2) and Rank biased overlap (RBO) [[3]](#3) of top-K features. With ranks $A$ and $B$, the Jaccard similarity is given by 
$$
    J(A,B)@k = \\frac{|A_k\cap B_k|}{|A_k\cup B_k|}. 
$$


where $A_k$ and $B_k$ contain only the top k features. Jaccard similarity and Kendall's Tau distance consider whether two sets contains the same elements rather than the consistency of ranking. And RBO is a weighted non-conjoint measure that focuses on the specific rankings of the elements, whcih is calculated as the average agreement of $A$ and $B$ of each depth:
$$
    RBO@k  = \\frac{1}{k} \sum_{d=1}^k \\frac{|A_d\cap B_d|}{d}
$$



#### Clustering Metrics

Clustering results are generated with the oracle number of clusters. We utilize the \textit{Adjusted Rand Index} (ARI) [[4]](#4), which is a widely used metric in clustering problem. The accuracy of clustering methods is measured by the ARI of clustering results with the true label, and the clustering consistency is calculated as the pairwise ARI of two clustering results. We also include additional common clustering metrics such mutual information [[5]](#5), V\_measure [[6]](#6) and fowlkes mallows [[7]](#7). 

#### Dimension Reduction Metrics 

One of the main goal of dimension reduction techniques is to draw inference from the visualization, which implies relative distances among the observations in the reduced dimension. Or researchers aim to conduct further downstream analysis such as clustering. Therefore, with focus on these two purposes, we measure the reliability of dimension reduction techniques in two directions: 

* K-nearest neighbor consistency in the reduced dimension

* Clustering consistency after dimension reduction

The K-nearest neighbor (KNN) consistency provides a quantitative consistency metric to exam the local similarity among given reduced dimensions. Given one specific observation, we can find its nearest neighbors of in each reduced dimension, and calculate the similarity of two sets of nearest neighbors by the Jaccard score [[1]](#1). Note that for $K = N$, the consistency measured by Jaccard score is always $1$ as the N-nearest neighbor contains the whole set of observations. The curve of Jaccard scores computed with $K = 1$ to $K=N$ against $K$ can be regarded as a receiver operating characteristic curve, we can further obtain area under the curve ($AUC$) of the Jaccard scores curve, denoted as $NN-Jaccard-AUC$ score. $NN-Jaccard-AUC  = 1$ indicates that the dimension reduction method is perfectly consistent in terms of local similarity under any $K$ range. Higher value of $NN-Jaccard-AUC$ indicates higher similarities between two reduced dimensions. 

On the other hand, we also measure the dimension reduction + clustering consistency by applying clustering algorithms to the reduced dimensions with oracle number of clusters. Following the same framework in the clustering consistency, we measure the accuracy of dimension reduction + clustering results by the ARI between true labels, and the consistency by pairwise ARI between clustering results. All of the metrics range from $[0,1]$, with large values indicating stronger consistency.


### Prediction Consistency Metrics in Supervised Learning 

If the training models are similar under different training sets, would they also generate similar interpretations? This question is specific to supervised learning methods, where we wish to explore the relations between the consistency of predicted estimates and the consistency of feature importance scores. Specifically, for each split, we build a prediction model using the training set, record the feature importance scores, and make predictions on the test set. Then we measure the prediction consistency of each sample by leveraging the dissimilarity of its predicted values. 

In the classification task, the dissimilarity is evaluated by the entropy of the predicted classification groups, where the entropy of sample $i$ is defined as 
$$
entropy_i = - \sum_{k=1}^{K} p_k \log(p_k) 
$$
where $K$ is the number of classes and $p_k$ is the proportion of class $k$. 

In the regression task, the standard deviation of predicted values is used as the consistency metric. In both scenarios, a smaller value indicates higher purity of the predicted responses, and we construct the consistency scores by 1 minus the average of min-max normalized dissimilarity values. 


### Data
#### 13 Classification Data Sets
|Data | # observations |   # features |  # classes | Type | Citation |
|:--- |    :----:      |    :----:    |     :----: |:----:|---:      |
|Bean | 13611 | 16 | 7 | images of grains of different registered dry beans | [[8]](#8)|
|Call | 7195 | 22 | 10 | acoustic features extracted from syllables of anuran (frogs) call | [[8]](#8)|
|Statlog|2310 | 19 | 7 | image segmentation database |[[8]](#8)|
|Theorem|6118|51|6|predict which of five heuristics will give the fastest proof when used by a first-order prover | [[9]](#9)|
|MNIST|70,000|784|10|digits image |[[10]](#10)|
|Spambase | 4601 | 57 | 2 |spam and non-spam email |[[8]](#8)|
|Author | 841 | 69 | 4 | word counts from chapters written by four famous English-language authors|[[8]](#8)|
|Amphibians|189|21|2|predict the presence of amphibians species|[[11]](#11)|
|Madelon | 2000 | 500 | 2 | artificial dataset | [[12]](#12)|
|TCGA | 445 | 353 | 5 | breast cancer data | [[13]](#13)|
|DNase | 386 | 2000 | 6 | high dimensional DNase      |[[14]](#14) | 
|Asian Religions | 590 | 8266 | 8 | characteristics of asian religion | [[15]](#15) |
|PANCAN | 761 | 13,244 | 5 | high dimensional RNA-seq |[[16]](#16)|

#### 13 Regression Data Sets
|Data |  # observations |  # features |  Type  | Citation |
|:--- |    :----:      |    :----:    |     :----: |:----:| 
|Online News Popularity | 39644 | 59 | predict \# of shares in social networks | [[17]](#17)|
|BlogFeedback |52397 |280 | predict how many comments the post will receive |[[18]](#18)|
|Satellite image|6435|36||[[19]](#19)|
|STAR | 2161 | 39 | Tennessee Student Teacher Achievement Ratio (STAR) project |[[20]](#20)|
|Communities and crime| 1993 | 99 | predict \# of violent crimes |[[8]](#8)|
|Bike|731|13|hourly and daily count of rental bikes| [[21]](#21)|
|CPU|209|7||[[19]](#19)|
|Wine|178|13|red wine quality|[[22]](#22)|
|Music|1059|117|geographical original of music |[[19]](#19)|
|Residential|372 | 103 |predict house price |[[23]](#23)|
|Tecator|240|124||[[19]](#19)|
|Word |523 | 526 | a word occurrence data to predict the length of a newsgroup record |[[8]](#8) |
|Riboflavin |71|4088 | genomics data set about riboflavin production rate|[[24]](#24) |


#### 14 Clustering Data Sets
|Data | # observations |# features |# classes |Type |Citation|
|:--- |    :----:      |    :----:  |    :----:    |     :----: |:----:| 
|Bean | 13611 | 16 | 7 | images of grains of different registered dry beans |[[8]](#8)|
|Call | 7195 | 22 | 10 | acoustic features extracted from syllables of anuran (frogs) call | [[8]](#8)|
|Statlog|2310 | 19 | 7 | image segmentation database |[[8]](#8)|
|Spambase | 4601 | 57 | 2 |spam and non-spam email|[[8]](#8)|
|Iris | 150 | 4  | 3  |types of iris plant |[[8]](#8)|
|WDBC| 569 | 30 | 2 |breast cancer wisconsin (diagnostic)  |[[8]](#8)|
|Tetragonula| 236 | 13| 9|Tetragonula bee species|[[25]](#25) |
|Author | 841 | 69 | 4 | word counts from chapters written by four famous English-language authors|[[8]](#8)|
|Ceramic | 88 | 17 | 2 |chemical composition of ceramic samples|[[8]](#8)|
|TCGA | 445 | 353 | 5 | breast cancer data | [[13]](#13)|
|Psychiatrist | 30 | 24  | 2 |case study psychiatrist| [[26]](#26)|
|Veronica | 206 | 583 | 8 | AFLP data of Veronica plants| [[27]](#27)|
|Asian Religions | 590 | 8266 | 8 | characteristics of asian religion | [[15]](#15) |
|PANCAN | 761 | 13,244 | 5 | high dimensional RNA-seq |[[16]](#16)|



#### 11 Dimension Reduction Data Sets
|Data | # observations |# features |# classes |Type |Citation|
|:--- |    :----:      |    :----:  |    :----:    |     :----: |:----:| 
|Statlog|2310 | 19 | 7 | image segmentation database |[[8]](#8)|
|Spambase | 4601 | 57 | 2 |spam and non-spam email|[[8]](#8)|
|WDBC| 569 | 30 | 2 |breast cancer wisconsin (diagnostic)  |[[8]](#8)|
|Tetragonula| 236 | 13 |tetragonula bee species|[[25]](#25) |
|Author | 841 | 69 | 4 | word counts from chapters written by four famous English-language authors|[[8]](#8)|
|TCGA | 445 | 353 | 5 | breast cancer data | [[13]](#13)|
|Veronica | 206 | 583 | 8 | AFLP data of Veronica plants| [[27]](#27)|
|Asian Religions | 590 | 8266 | 8 | characteristics of asian religion | [[15]](#15) |
|PANCAN | 761 | 13,244 | 5 | high dimensional RNA-seq |[[16]](#16)|
|Darmanis | 366 |21,413 | 4 |  high dimensional brain single cell RNA-seq|[[28]](#28)|



## References
<a id="1">[1]</a> 
Real, Raimundo, and Juan M. Vargas. "The probabilistic basis of Jaccard's index of similarity." Systematic biology 45.3 (1996): 380-385.


<a id="2">[2]</a> 
Kendall, Maurice G. "A new measure of rank correlation." Biometrika 30.1/2 (1938): 81-93.



<a id="3">[3]</a> 
Webber, William, Alistair Moffat, and Justin Zobel. "A similarity measure for indefinite rankings." ACM Transactions on Information Systems (TOIS) 28.4 (2010): 1-38.

<a id="4">[4]</a> 
Rand, William M. "Objective criteria for the evaluation of clustering methods." Journal of the American Statistical association 66.336 (1971): 846-850.


<a id="5">[5]</a> 
Kraskov, Alexander, Harald Stögbauer, and Peter Grassberger. "Estimating mutual information." Physical review E 69.6 (2004): 066138.

<a id="6">[6]</a> 
Rosenberg, Andrew, and Julia Hirschberg. "V-measure: A conditional entropy-based external cluster evaluation measure." Proceedings of the 2007 joint conference on empirical methods in natural language processing and computational natural language learning (EMNLP-CoNLL). 2007.


<a id="7">[7]</a> 
Fowlkes, Edward B., and Colin L. Mallows. "A method for comparing two hierarchical clusterings." Journal of the American statistical association 78.383 (1983): 553-569.



<a id="8">[8]</a> 
Blake, Catherine. 
UCI repository of machine learning databases.
http://www. ics. uci. edu/~ mlearn/MLRepository. html (1998).

<a id="9">[9]</a> 
Bridge, James P., Sean B. Holden, and Lawrence C. Paulson. 
Machine learning for first-order theorem proving: learning to select a good heuristic.
Journal of automated reasoning 53 (2014): 141-172.


<a id="10">[10]</a> 
Deng, Li. "The mnist database of handwritten digit images for machine learning research [best of the web]." IEEE signal processing magazine 29.6 (2012): 141-142.

<a id="11">[11]</a> 
Habib, Nadia Shaker, et al. 
Presence of Amphibian Species Prediction Using Features Obtained from GIS and Satellite Images.
International Journal of Academic and Applied Research (IJAAR) 4.11 (2020).

<a id="12">[12]</a> 
Guyon, Isabelle, et al. 
Result analysis of the nips 2003 feature selection challenge.
Advances in neural information processing systems 17 (2004).

<a id="13">[13]</a> 
Brigham & Women’s Hospital & Harvard Medical School Chin Lynda 9 11 Park Peter J. 12 Kucherlapati Raju 13, et al. 
Comprehensive molecular portraits of human breast tumours.
Nature 490.7418 (2012): 61-70.

<a id="14">[14]</a> 
ENCODE Project Consortium. 
An integrated encyclopedia of DNA elements in the human genome.
Nature 489.7414 (2012): 57.



<a id="15">[15]</a> 
Sah, Preeti, and Ernest Fokoué. 
What do asian religions have in common? an unsupervised text analytics exploration.
arXiv preprint arXiv:1912.10847 (2019).

<a id="16">[16]</a> 
Weinstein, John N., et al. 
The cancer genome atlas pan-cancer analysis project.
Nature genetics 45.10 (2013): 1113-1120.

<a id="17">[17]</a> 
Fernandes, Kelwin, Pedro Vinagre, and Paulo Cortez. 
A proactive intelligent decision support system for predicting the popularity of online news.

<a id="18">[18]</a> 
Buza, Krisztian. 
Feedback prediction for blogs.
Data analysis, machine learning and knowledge discovery. Springer, Cham, 2014. 145-152.

<a id="19">[19]</a> 
Romano, Joseph D., et al. "PMLB v1. 0: an open-source dataset collection for benchmarking machine learning methods." Bioinformatics 38.3 (2022): 878-880.


<a id="20">[20]</a> 
C.M. Achilles; Helen Pate Bain; Fred Bellott; Jayne Boyd-Zaharias; Jeremy Finn; John Folger; John Johnston; Elizabeth Word, 2008, "Tennessee's Student Teacher Achievement Ratio (STAR) project", https://doi.org/10.7910/DVN/SIWH9F, Harvard Dataverse, V1, UNF:3:Ji2Q+9HCCZAbw3csOdMNdA== [fileUNF]


<a id="21">[21]</a> 
Fanaee-T, Hadi, and Joao Gama. "Event labeling combining ensemble detectors and background knowledge." Progress in Artificial Intelligence 2 (2014): 113-127.


<a id="22">[22]</a> 
Cortez, Paulo, et al. "Modeling wine preferences by data mining from physicochemical properties." Decision support systems 47.4 (2009): 547-553.
<a id="4">[4]</a> 
Brigham & Women’s Hospital & Harvard Medical School Chin Lynda 9 11 Park Peter J. 12 Kucherlapati Raju 13, et al. 
Comprehensive molecular portraits of human breast tumours.
Nature 490.7418 (2012): 61-70.

<a id="23">[23]</a> 
Rafiei, Mohammad Hossein, and Hojjat Adeli. 
A novel machine learning model for estimation of sale prices of real estate units.
Journal of Construction Engineering and Management 142.2 (2016): 04015066.


<a id="24">[24]</a> 
Bühlmann, Peter, Markus Kalisch, and Lukas Meier. 
High-dimensional statistics with a view toward applications in biology.
Annual Review of Statistics and Its Application 1.1 (2014): 255-278.Result analysis of the nips 2003 feature selection challenge.


<a id="25">[25]</a> 
Franck, Pierre, et al. "Nest architecture and genetic differentiation in a species complex of Australian stingless bees." Molecular Ecology 13.8 (2004): 2317-2331.


<a id="26">[26]</a> 
Van Mechelen, Iven, and Paul De Boeck. "Implicit taxonomy in psychiatric diagnosis: A case study." Journal of Social and Clinical Psychology 8.3 (1989): 276-287.

<a id="27">[27]</a>
Martínez-Ortega, M. Montserrat, et al. "Species boundaries and phylogeographic patterns in cryptic taxa inferred from AFLP markers: Veronica subgen. Pentasepalae (Scrophulariaceae) in the Western Mediterranean." Systematic Botany 29.4 (2004): 965-986.

<a id="28">[28]</a> 
Darmanis, Spyros, et al. "A survey of human brain transcriptome diversity at the single cell level." Proceedings of the National Academy of Sciences 112.23 (2015): 7285-7290.







''',
    mathjax=True, dangerously_allow_html=True, style={'marginLeft': '5%','marginRigt': '5%', 'width': '90%'})

# design = 'fig/design.pdf' # replace with your own image
# # f1_image = base64.b64encode(open(f1, 'rb').read())


def Homepage():
    layout = html.Div([
    nav,
    body,
    content
        
        
    ])
    return layout

# app = dash.Dash(__name__, external_stylesheets = [dbc.themes.UNITED])
# app.layout = Homepage()
# if __name__ == "__main__":
#     app.run_server()
    