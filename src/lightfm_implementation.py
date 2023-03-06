# Reference: https://github.com/yuwei-jacque-wang/Recommender-System-DSGA1004/blob/master/LightFM.ipynb

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import lightfm
from lightfm.data import Dataset

import json
from itertools import islice

import pandas as pd
import numpy as np
from collections import Counter

from scipy.sparse import csr_matrix

from lightfm import LightFM  

from lightfm.cross_validation import random_train_test_split

from lightfm.evaluation import precision_at_k       
from time import time


def transform_interaction(df, test_percent):
    
    interaction = pd.pivot_table(df, index='userId', columns='movieId', values='rating')
    interaction = interaction.fillna(0)
    
    all_csr = csr_matrix(interaction.values)
    
    (train_matrix, test_matrix) = random_train_test_split(all_csr, test_percentage=test_percent)
    
    return (train_matrix, test_matrix)


def lightfm_train(train, rank, regParam, maxIter, model_type='warp'):

    if model_type == 'bpr':
        model = LightFM(loss='bpr',
                no_components=rank,
                user_alpha=regParam)
        
    else:    
        model = LightFM(loss='warp',
                no_components=rank,
                user_alpha=regParam)

    model = model.fit(train, epochs=maxIter,verbose=False)
    
    return model



def train_and_test(train, test, rank, regParam, maxIter, top=500, model_type='warp'):    
        
    st = time()
    
    model = lightfm_train(train, rank, regParam, maxIter, model_type='warp')
    p_at_k = precision_at_k(model, test, k=top).mean()
    
    t = round(time()-st, 5)
    
    print('Model with maxIter = {}, reg = {}, rank = {} complete'.format(maxIter,regParam,rank))
    print('Precision at K:', p_at_k)
    print('Time used:', t)
    
    output = "Model with maxIter =" + str(maxIter) +', reg = ' + str(regParam) + ', rank =' + str(rank) + ' complete.\nPrecision at K:' + str(p_at_k) +'\nTime used:' + str(t)
    
    return output


train = pd.read_csv('../data/train_data_small.csv')
val = pd.read_csv('../data/val_data_small.csv')
test = pd.read_csv('../data/test_data_small.csv')

raw = pd.concat([train, val, test])
raw = raw.drop(['timestamp','title','genres'],axis=1)


train_1, test_1 = transform_interaction(raw, 0.2)

with open('../results/lightfm_warp_rankResults.txt','w') as f:
    for rank in [10,20,30,40,50,60,70,80,90,100,120,140,160,180]:
        result = train_and_test(train_1, test_1, rank, 0.01, 10, top=500, model_type='warp')
        f.write(result+'\n')


with open('../results/lightfm_bpr_rankResults.txt','w') as f:
    for rank in [10,20,30,40,50,60,70,80,90,100,120,140,160,180]:
        result = train_and_test(train_1, test_1, rank, 0.01, 10, top=500, model_type='bpr')
        f.write(result+'\n')


with open('../results/lightfm_warp_paramResults.txt','w') as f:
    for regParam in [0.001, 0.01, 0.05, 0.1, 0.5]:
        result = train_and_test(train_1, test_1, 160, regParam, 10, top=500, model_type='warp')
        f.write(result+'\n')


with open('../results/lightfm_bpr_paramResults.txt','w') as f:
    for regParam in [0.001, 0.01, 0.05, 0.1, 0.5]:
        result = train_and_test(train_1, test_1, 160, regParam, 10, top=500, model_type='bpr')
        f.write(result+'\n')
