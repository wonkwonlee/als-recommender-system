# Collaborative-filter Based Recommender System

Collaborative-filtering Movie Recommender System using Spark's Alternating Least Squares (ALS) method

Final Project of Big Data - DS.GA.1004 Spring 2022


## Dataset
[MovieLens](https://grouplens.org/datasets/movielens/latest/) datasets 
 - F. Maxwell Harper and Joseph A. Konstan. 2015. 
 - The MovieLens Datasets: History and Context. 
 - ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19
 https://doi.org/10.1145/2827872

## Method
- ALS method is a matrix factorization algorithm that decomposes the user-item interaction matrix into the product of two lower dimensionality rectangular matrices, user factor matrix U and item factor matrix V
- Parallel ALS method is compared to a single-machine implmentation using LightFM


## Evaluation
- MAP: Mean Average Precision
- MAP @ K: Mean Average Precision of top K recommended items
- NDCG: Normalized Discounted Cumulative Gain of top K recommended items