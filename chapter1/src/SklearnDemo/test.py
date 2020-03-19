#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   test.py    
@Contact :   tianze05@163.com
@License :   (C)Copyright 2020-2030, LT.CT

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/3/12 23:45   tazawa      1.0         None
'''

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler,StandardScaler

tranf = MinMaxScaler(feature_range=(0,1))
data_new = tranf.fit_transform()
