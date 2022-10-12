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
    J(A,B)@k = \frac{|A_k\cap B_k|}{|A_k\cup B_k|}. 
$$

where $A_k$ and $B_k$ contain only the top k features. However, the Jaccard similarity considers whether two sets contains the same elements rather than the consistency of ranking. Hence, we also propose another metric Rank biased overlap (RBO) \citet{webber2010similarity}, which is a weighted non-conjoint measure that focuses on the specific rankings of the elements. Specifically, RBO is calculated as the average agreement of $A$ and $B$ of each depth and is given by
$$
    RBO@k  = \frac{1}{k} \sum_d=1^k \frac{|A_d\cap B_d|}{d}
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
#### Classification Data Sets
|Data | # observations |   # features |  # classes | Type | Citation |
|:--- |    :----:      |    :----:    |     :----: |:----:|---:      |
|Asian Religions | 590 | 8266 | 8 |    | \cite{sah2019asian} |
|PANCAN | 761 | 13,244 | 5 | high dimensional RNA-seq | \citet{weinstein2013cancer}|
|DNase | 386 | 2000 | 6 | high dimensional DNase      | \citet{encode2012integrated} | 
|madelon | 2000 | 500 | 2 | artificial dataset | \citet{guyon2004result}|
|Spambase | 4601 | 57 | 2 |Classifying Email as Spam or Non-Spam | \citet{} |
|Author | 841 | 69 | 4 | word counts from chapters written by four famous
|Bean | 13611 | 16 | 7 | Images of 13,611 grains of 7 different registered dry beans 

#### Regression Data Sets
|Data |  # observations |  # features |  Type  | Citation |
|:--- |    :----:      |    :----:    |     :----: |:----:| 
|Riboflavin |71|4088 | genomics data set about riboflavin production rate|\cite{buhlmann2014high}|
|RNA-seq | 475 | 48803 | high dimensional, from ROSMAP, biological data | \citet{} |
|BlogFeedback |52397 |280 |to predict how many comments the post will receive | \citet{buza2014feedback} |
|Online News Popularity | 39644 | 59 | predict \# of shares in social networks | \citet{fernandes2015proactive} |
|STAR | 2161 | 39 | Tennessee Student Teacher Achievement Ratio (STAR) project | \citet{DVN/SIWH9F_2008}|
|Word |523 | 526 | binary; A word occurrence data to predict the length of a newsgroup record | \citet{} |
|Communities and crime| 1993 | 99 | predict \# of violent crimes | \citet{}|


#### Clustering & Dimension Reduction Data Sets
|Data | # observations |# features |# classes |Type |Citation|
|:--- |    :----:      |    :----:  |    :----:    |     :----: |:----:| 
|PANCAN | 761 | 13,244 | 5 | high dimensional RNA-seq | \citet{weinstein2013cancer} |
|DNase | 386 | 2000 | 30 | high dimensional DNase | \citet{encode2012integrated}| 
|Asian Religions | 590 | 8266 | 8 | | \cite{sah2019asian}|
|Author | 841 | 69 | 4 | word counts from chapters written by four famous
|Spambase | 4601 | 57 | 2 |Classifying Email as Spam or Non-Spam | \citet{} |
|statlog | 2310 | 19 | 7 | image segmentation database | \citet{} |
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
    