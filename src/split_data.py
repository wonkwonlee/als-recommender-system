#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split

print("Reading in ratings.csv")
df_rating = pd.read_csv("/scratch/work/courses/DSGA1004-2021/movielens/ml-latest-small/ratings.csv")
print(df_rating.head())

print("\nReading in movies.csv") 
df_movies = pd.read_csv("/scratch/work/courses/DSGA1004-2021/movielens/ml-latest-small/movies.csv")  
print(df_movies.head())

print("\nMerging data")
df = pd.merge(df_rating, df_movies, on="movieId")
print(df.head())
print("\nDataframe length:",len(df))

 
train, test_data = train_test_split(df, test_size = 0.15)
train_data, val_data = train_test_split(train, test_size = 0.15)

print("\nTrain data length:", len(train_data))
print("Test data length:", len(test_data))
print("Validation data length:", len(val_data)) 


print("\nExporting split datasets")
train_data.to_csv(r'../data/train_data.csv',index=False)
test_data.to_csv(r'../data/test_data.csv',index=False)
val_data.to_csv(r'../data/val_data.csv',index=False)


print("Done!")  
