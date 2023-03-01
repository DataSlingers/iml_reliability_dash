##### import packages 
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

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


### IML tasks
We focus on three major types machine learning tasks, including: 

* Feature Importance 
* Clustering (interpretability of observations groups)
* Dimension Reduction (provide interpretations of observations patterns)
### Sensitivity Tests

   We define the interpretability to be reliable if the same or similar interpretations can be derived from new data of the same distribution. Therefore, the reliability of a machine learning model can be measured by the consistency of its derived interpretations, and a reliable machine learning model should not be over over-sensitive to small changes in the data or parameters of the model. To this end, the first step of our framework is to design sensitivity tests to obtain interpretations from machine learning models under different circumstance. 

   We have two types of sensitivity test:

   * Random train/test splits
   * Noise addition. 

   The random train/test splits is used for both supervised and unsupervised models. For example, the predictive model has reliable interpretability if the resulting feature importance scores can remain unchanged and consistent with different random train/test splits. Moreover, in unsupervised learning, the consistency of interpretations and its resistance to additional noise can be obtained by adding random noise to the data of interest. With increasing levels of noise added, we are able to illustrate how the consistency would change with more difficult data. Additionally, we may also test how the interpretations changes with different parameter settings.  

### Interpretation Consistency Metrics
We include three categories of interpretabilities: 1). feature importance/ranking derived from supervised learning; 2). clustering results, which implies underlying connections among observations; 3). interpretations in reduced dimension. In each category, we develop robust metrics to measure the reliability of the resulting interpretations. 

#### Feature Importance Metrics
As researcher are generally more interested in the most important features, we focus on top-K rank consistency in the case of feature importance. One of the metric is the Jaccard similarity \citep{real1996probabilistic} of top-K features. With ranks $A$ and $B$, the Jaccard similarity is given by 
$$
    J(A,B)@k = \\frac{|A_k\cap B_k|}{|A_k\cup B_k|}. 
$$

where $A_k$ and $B_k$ contain only the top k features. However, the Jaccard similarity considers whether two sets contains the same elements rather than the consistency of ranking. Hence, we also propose another metric Rank biased overlap (RBO) \citet{webber2010similarity}, which is a weighted non-conjoint measure that focuses on the specific rankings of the elements. Specifically, RBO is calculated as the average agreement of $A$ and $B$ of each depth and is given by
$$
    RBO@k  = \\frac{1}{k} \sum_{d=1}^k \\frac{|A_d\cap B_d|}{d}
$$



#### Clustering Metrics

Clustering results are generated with the oracle number of clusters. We utilize the \textit{Adjusted Rand Index} (ARI) \citep{rand1971objective}, which is a widely used metric in clustering problem. The accuracy of clustering methods is measured by the ARI of clustering results with the true label, and the clustering consistency is calculated as the pairwise ARI of two clustering results. We also include additional common clustering metrics such mutual information \citep{kraskov2004estimating}, V\_measure \citep{rosenberg2007v} and fowlkes mallows \citep{fowlkes1983method}

#### Dimension Reduction Metrics 

One of the main goal of dimension reduction techniques is to draw inference from the visualization, which implies relative distances among the observations in the reduced dimension. Or researchers aim to conduct further downstream analysis such as clustering. Therefore, with focus on these two purposes, we measure the reliability of dimension reduction techniques in two directions: 
* K-nearest neighbor consistency in the reduced dimension

* Clustering consistency after dimension reduction

The K-nearest neighbor (KNN) consistency provides a quantitative consistency metric to exam the local similarity among given reduced dimensions. Given one specific observation, we can find its nearest neighbors of in each reduced dimension, and calculate the similarity of two sets of nearest neighbors by the Jaccard score \citep{real1996probabilistic}. Note that for $K = N$, the consistency measured by Jaccard score is always $1$ as the N-nearest neighbor contains the whole set of observations. The curve of Jaccard scores computed with $K = 1$ to $K=N$ against $K$ can be regarded as a receiver operating characteristic curve, we can further obtain area under the curve ($AUC$) of the Jaccard scores curve. $AUC  = 1$ indicates that the dimension reduction method is perfectly consistent in terms of local similarity under any $K$ range. Higher value of $AUC$ indicates higher similarities between two reduced dimensions. 

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

|Bean | 13611 | 16 | 7 | Images of 13,611 grains of 7 different registered dry beans| [[5]](#5) |
|Call | 7195 | 22 | 10 | Acoustic features extracted from syllables of anuran (frogs) call | |
|Statlog|2310 | 19 | 7 | image segmentation database | \citet{} |

|Theorem|6118|51|6|predict which of five heuristics will give the fastest proof when used by a first-order prover | \citet{bridge2014machine}|
|MNIST|70,000|784|10|Digits image | \citet{deng2012mnist}|
|Spambase | 4601 | 57 | 2 |Classifying Email as Spam or Non-Spam | [[5]](#5) |
|Author | 841 | 69 | 4 | word counts from chapters written by four famous English-language authors| [[5]](#5) |
|Amphibians|189|21|2|predict the presence of amphibians species&\citet{habib2020presence}|
|Madelon | 2000 | 500 | 2 | artificial dataset | [[4]](#4)|
|TCGA Breast Cancer Data | 445 | 353 | 5 | | \citet{koboldt2012comprehensive}|

|DNase | 386 | 2000 | 6 | high dimensional DNase      |[[3]](#3) | 

|Asian Religions | 590 | 8266 | 8 |    | [[1]](#1) |
|PANCAN | 761 | 13,244 | 5 | high dimensional RNA-seq |[[2]](#2)|

#### 13 Regression Data Sets
|Data |  # observations |  # features |  Type  | Citation |
|:--- |    :----:      |    :----:    |     :----: |:----:| 
|Online News Popularity | 39644 | 59 | predict \# of shares in social networks | [[9]](#9)|
|BlogFeedback |52397 |280 |to predict how many comments the post will receive |[[8]](#8)|
|Satellite image|6435|36||\citet{romano2021pmlb}|
|STAR | 2161 | 39 | Tennessee Student Teacher Achievement Ratio (STAR) project |[[10]](#10)|
|Communities and crime| 1993 | 99 | predict \# of violent crimes |  [[5]](#5) |
|Bike|731|13|hourly and daily count of rental bikes| \citet{fanaee2014event}|
|CPU|209|7||\citet{romano2021pmlb}|
|Wine|178|13|red wine quality|\citet{cortez2009modeling}|
|Music|1059|117|Geographical Original of Music |\citet{romano2021pmlb}|
|Residential|372 | 103 |predict house price | \citet{rafiei2016novel}|
|Tecator|240|124||\citet{romano2021pmlb}|
|Word |523 | 526 | binary; A word occurrence data to predict the length of a newsgroup record | |
|Riboflavin |71|4088 | genomics data set about riboflavin production rate|[[6]](#6) |


#### 14 Clustering Data Sets
|Data | # observations |# features |# classes |Type |Citation|
|:--- |    :----:      |    :----:  |    :----:    |     :----: |:----:| 
|Bean | 13611 | 16 | 7 | Images of 13,611 grains of 7 different registered dry beans| [[5]](#5) |
|Call | 7195 | 22 | 10 | Acoustic features extracted from syllables of anuran (frogs) call | |
|Statlog|2310 | 19 | 7 | image segmentation database | \citet{} |
|Spambase | 4601 | 57 | 2 |Classifying Email as Spam or Non-Spam | [[5]](#5) |
|Iris | 150 | 4  | 3  ||  |
|WDBC| 569 | 30 | 2 |||
|Tetragonula| 236 | 13 |||
|Author | 841 | 69 | 4 | word counts from chapters written by four famous English-language authors| [[5]](#5) |
|Ceramic | 88 | 17 | 2 |||
|TCGA Breast Cancer Data | 445 | 353 | 5 | | \citet{koboldt2012comprehensive}|
|Psychiatrist | 30 | 24  | 2 |||
|Veronica | 206 | 583 | 8 |||
|Asian Religions | 590 | 8266 | 8 |    | [[1]](#1) |
|PANCAN | 761 | 13,244 | 5 | high dimensional RNA-seq |[[2]](#2)|



#### 12 Dimension Reduction Data Sets
|Data | # observations |# features |# classes |Type |Citation|
|:--- |    :----:      |    :----:  |    :----:    |     :----: |:----:| 
|Statlog|2310 | 19 | 7 | image segmentation database | \citet{} |
|Spambase | 4601 | 57 | 2 |Classifying Email as Spam or Non-Spam | [[5]](#5) |
|WDBC| 569 | 30 | 2 |||
|Tetragonula| 236 | 13 |||
|Author | 841 | 69 | 4 | word counts from chapters written by four famous English-language authors| [[5]](#5) |
|Madelon | 2000 | 500 | 2 | artificial dataset | [[4]](#4)|
|TCGA Breast Cancer Data | 445 | 353 | 5 | | \citet{koboldt2012comprehensive}|
|Psychiatrist | 30 | 24  | 2 |||
|Veronica | 206 | 583 | 8 |||
|Asian Religions | 590 | 8266 | 8 |    | [[1]](#1) |
|PANCAN | 761 | 13,244 | 5 | high dimensional RNA-seq |[[2]](#2)|
|Darmanis | 366 |21,413 | 4 |  high dimensional brain single cell RNA-seq| \citet{darmanis2015survey}|



## References
<a id="1">[1]</a> 
Sah, Preeti, and Ernest Fokoué. 
What do asian religions have in common? an unsupervised text analytics exploration.
arXiv preprint arXiv:1912.10847 (2019).

<a id="2">[2]</a> 
Weinstein, John N., et al. 
The cancer genome atlas pan-cancer analysis project.
Nature genetics 45.10 (2013): 1113-1120.

<a id="3">[3]</a> 
ENCODE Project Consortium. 
An integrated encyclopedia of DNA elements in the human genome.
Nature 489.7414 (2012): 57.

<a id="4">[4]</a> 
Brigham & Women’s Hospital & Harvard Medical School Chin Lynda 9 11 Park Peter J. 12 Kucherlapati Raju 13, et al. 
Comprehensive molecular portraits of human breast tumours.
Nature 490.7418 (2012): 61-70.


<a id="5">[5]</a> 
Guyon, Isabelle, et al. 
Result analysis of the nips 2003 feature selection challenge.
Advances in neural information processing systems 17 (2004).

<a id="6">[6]</a> 
Blake, Catherine. 
UCI repository of machine learning databases.
http://www. ics. uci. edu/~ mlearn/MLRepository. html (1998).



<a id="7">[7]</a> 
Bridge, James P., Sean B. Holden, and Lawrence C. Paulson. 
Machine learning for first-order theorem proving: learning to select a good heuristic.
Journal of automated reasoning 53 (2014): 141-172.



<a id="8">[8]</a> 
Habib, Nadia Shaker, et al. 
Presence of Amphibian Species Prediction Using Features Obtained from GIS and Satellite Images.
International Journal of Academic and Applied Research (IJAAR) 4.11 (2020).





<a id="6">[6]</a> 
Bühlmann, Peter, Markus Kalisch, and Lukas Meier. 
High-dimensional statistics with a view toward applications in biology.
Annual Review of Statistics and Its Application 1.1 (2014): 255-278.Result analysis of the nips 2003 feature selection challenge.


Bennett, David A., et al. 
Religious orders study and rush memory and aging project. 
Journal of Alzheimer's disease 64.s1 (2018): S161-S189.

Buza, Krisztian. 
Feedback prediction for blogs.
Data analysis, machine learning and knowledge discovery. Springer, Cham, 2014. 145-152.

<a id="9">[9]</a> 
Fernandes, Kelwin, Pedro Vinagre, and Paulo Cortez. 
A proactive intelligent decision support system for predicting the popularity of online news.
Portuguese conference on artificial intelligence. Springer, Cham, 2015.

<a id="10">[10]</a> 
Word, Elizabeth R. 
The State of Tennessee's Student/Teacher Achievement Ratio (STAR) Project: Technical Report (1985-1990).(1990).


<a id="11">[11]</a> 
Sah, Preeti, and Ernest Fokoué.
What do asian religions have in common? an unsupervised text analytics exploration. 
arXiv preprint arXiv:1912.10847 (2019).

Rafiei, Mohammad Hossein, and Hojjat Adeli. 
A novel machine learning model for estimation of sale prices of real estate units.
Journal of Construction Engineering and Management 142.2 (2016): 04015066.


''',
    mathjax=True, dangerously_allow_html=True, style={'marginLeft': '10%', 'width': '90%'})


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
    