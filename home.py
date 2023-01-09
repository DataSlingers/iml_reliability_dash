##### import packages 
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

from nav import *
#from ipynb.fs.full.navbar import Navbar
nav = Navbar()
body = dbc.Container([
       html.H1('Can We Trust Machine Learning Interpretations? A Reliability Study'),
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
className="mt-4",
)
content= dcc.Markdown(
    
'''
### Goal
   **We aim to design a clear large-scale empirical framework to evaluate the reliability of interpretabilities of popular interpretable machine learning(IML) models. **
### Contribution
  Through this project, we make the following constributions:

* Designed a clear large-scale framework to evaluate the reliability of interpretability of popular interpretable machine learning (IML) models 
* Incorporated rigorous empirical analysis and developing reliable metrics via sensitivity tests
* Implemented the aforementioned framework and evaluated the interpretability of 50+ IML & deep learning models in the fields of both supervised (e.g., random forest, SVM, MLP) and unsupervised learnings (e.g., autoencoder, PCA, t-SNE)

### This Dashboard

This dashboard provides an interactive platform to present our results. In addition, we provide functions for users to evaluate their own data set and/or IML methods of interests. 


### IML tasks
We focus on three major types machine learning tasks, including: 

* Feature Importance 
* Clustering (interpretability of observations groups)
* Dimension Reduction (provide interpretations of observations patterns)
### Sensitivity Tests

   We define the interpretability to be reliable if the same or similar interpretations can be derived from new data of the same distribution. Therefore, the reliability of a machine learning model can be measured by the consistency of its derived interpretations.  For example, in supervised learning, researchers usually conduct train/test split before fitting a predictive model so as to avoid overfitting. The predictive model has reliable interpretability if the resulting feature importance scores can remain unchanged and consistent with different random train/test splits. Therefore, a reliable machine learning model should not be over over-sensitive to small changes in the data or parameters of the model. To this end, the first step of our framework is to design sensitivity tests to obtain interpretations from machine learning models under different circumstance. 

   We have two types of sensitivity test:

   * Random train/test splits
   * Noise addition. 

   The random train/test splits is used sorely for supervised models. The consistency of interpretations and its resistance to additional noise can be obtained by adding random noise to the data of interest. In addition, with increasing levels of noise added, we are able to illustrate how the consistency would change with more difficult data. Additionally, we may also test how the interpretations changes with different parameter settings.  

### Metrics of reliability 
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


### Data
#### 13 Classification Data Sets
|Data | # observations |   # features |  # classes | Type | Citation |
|:--- |    :----:      |    :----:    |     :----: |:----:|---:      |
|Asian Religions | 590 | 8266 | 8 |    | [[1]](#1) |
|PANCAN | 761 | 13,244 | 5 | high dimensional RNA-seq |[[2]](#2)|
|DNase | 386 | 2000 | 6 | high dimensional DNase      |[[3]](#3) | 
|TCGA Breast Cancer Data | 445 | 353 | 5 | | \citet{koboldt2012comprehensive}|
|Madelon | 2000 | 500 | 2 | artificial dataset | [[4]](#4)|
|Spambase | 4601 | 57 | 2 |Classifying Email as Spam or Non-Spam | [[5]](#5) |
|Author | 841 | 69 | 4 | word counts from chapters written by four famous English-language authors| [[5]](#5) |
|Bean | 13611 | 16 | 7 | Images of 13,611 grains of 7 different registered dry beans| [[5]](#5) |
|Call | 7195 | 22 | 10 | Acoustic features extracted from syllables of anuran (frogs) call | |
|Digit | 70,000 | 784 | 10 |image & \citet{deng2012mnist} |

|Theorem|6118|51|6|predict which of five heuristics will give the fastest proof when used by a first-order prover | \citet{bridge2014machine}|
|Statlog|2310 | 19 | 7 | image segmentation database | \citet{} |

|Amphibians|189|21|2|predict the presence of amphibians species&\citet{habib2020presence}|

#### 13 Regression Data Sets
|Data |  # observations |  # features |  Type  | Citation |
|:--- |    :----:      |    :----:    |     :----: |:----:| 
|Riboflavin |71|4088 | genomics data set about riboflavin production rate|[[6]](#6) |
|RNA-seq | 475 | 48803 | high dimensional, from ROSMAP, biological data | [[7]](#7) |
|BlogFeedback |52397 |280 |to predict how many comments the post will receive |[[8]](#8)|
|Online News Popularity | 39644 | 59 | predict \# of shares in social networks | [[9]](#9)|
|STAR | 2161 | 39 | Tennessee Student Teacher Achievement Ratio (STAR) project |[[10]](#10)|
|Word |523 | 526 | binary; A word occurrence data to predict the length of a newsgroup record | |
|Communities and crime| 1993 | 99 | predict \# of violent crimes |  [[5]](#5) |

|residential|372 | 103 |predict house price | \citet{rafiei2016novel}|
|bike|731|13|hourly and daily count of rental bikes| \citet{fanaee2014event}\\

|wine|178|13|red wine quality|\citet{cortez2009modeling}|
|music|1059|117|Geographical Original of Music |\citet{romano2021pmlb}|
|tecator|240|124||\citet{romano2021pmlb}|
|satellite image|6435|36||\citet{romano2021pmlb}|

|cpu|209&7&&\citet{romano2021pmlb}\\
#### Clustering & Dimension Reduction Data Sets
|Data | # observations |# features |# classes |Type |Citation|
|:--- |    :----:      |    :----:  |    :----:    |     :----: |:----:| 
|Data | # observations |   # features |  # classes | Type | Citation |
|:--- |    :----:      |    :----:    |     :----: |:----:|---:      |
|Asian Religions | 590 | 8266 | 8 |    | [[1]](#1) |
|PANCAN | 761 | 13,244 | 5 | high dimensional RNA-seq |[[2]](#2)|
|DNase | 386 | 2000 | 6 | high dimensional DNase      |[[3]](#3) | 
|TCGA Breast Cancer Data | 445 | 353 | 5 | | \citet{koboldt2012comprehensive}|
|Madelon | 2000 | 500 | 2 | artificial dataset | [[4]](#4)|
|Spambase | 4601 | 57 | 2 |Classifying Email as Spam or Non-Spam | [[5]](#5) |
|Author | 841 | 69 | 4 | word counts from chapters written by four famous English-language authors| [[5]](#5) |
|Bean | 13611 | 16 | 7 | Images of 13,611 grains of 7 different registered dry beans| [[5]](#5) |
|Call | 7195 | 22 | 10 | Acoustic features extracted from syllables of anuran (frogs) call | |
|Theorem|6118|51|6|predict which of five heuristics will give the fastest proof when used by a first-order prover | \citet{bridge2014machine}|
|Statlog|2310 | 19 | 7 | image segmentation database | \citet{} |

|Amphibians|189|21|2|predict the presence of amphibians species&\citet{habib2020presence}|



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
Guyon, Isabelle, et al. 
Result analysis of the nips 2003 feature selection challenge.
Advances in neural information processing systems 17 (2004).

<a id="5">[5]</a> 
Blake, Catherine. 
UCI repository of machine learning databases.
http://www. ics. uci. edu/~ mlearn/MLRepository. html (1998).



<a id="6">[6]</a> 
Bühlmann, Peter, Markus Kalisch, and Lukas Meier. 
High-dimensional statistics with a view toward applications in biology.
Annual Review of Statistics and Its Application 1.1 (2014): 255-278.Result analysis of the nips 2003 feature selection challenge.


<a id="7">[7]</a> 
Bennett, David A., et al. 
Religious orders study and rush memory and aging project. 
Journal of Alzheimer's disease 64.s1 (2018): S161-S189.

<a id="8">[8]</a> 
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

''',
    mathjax=True, dangerously_allow_html=True, style={'marginLeft': '5%', 'width': '90%'})


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
    