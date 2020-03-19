#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   sklearn_002.py    
@Contact :   tianze05@163.com
@License :   (C)Copyright 2020-2030, LT.CT

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/3/12 22:38   tazawa      1.0         特征预处理
'''
#归一化
"""
sklearn.preprocessing.MinMaxScaler(feature_range(0,1)...)
    MinMaxScaler.fit_transform(X)
        X:numpy array 格式的数据[n_samples,n_features]
    返回值：转换后的形状相同的array
"""
#归一化标准
"""
sklearn.preprocessing.StandardScaler()
    处理之后对每列来说，所有数据都聚集在均值为0附近，标准差为1
    MinMaxScaler.fit_transform(X)
        X:numpy array 格式的数据[n_samples,n_features]
    返回值：转换后的形状相同的array
"""
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import pandas as pd

def minmax_dema():
    """
    归一化
    :return:
    """
    # 1：获取数据
    filepath=r'E:\Python\txt\GP\HB\gpmx.xls'
    data = pd.read_excel(filepath,encoding='utf-8',sheet_name=0)
    data = data.iloc[:,2:6]
    # 2：实例化一个转换器类
    transfer = MinMaxScaler(feature_range=(0,1))
    # 3：调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n",data_new)
    return  None
def stand_demo():
    """
    归一化标准化
    :return: -=-
    """
    # 1：获取数据
    filepath = r'E:\Python\txt\GP\HB\gpmx.xls'
    data = pd.read_excel(filepath, encoding='utf-8', sheet_name=0)
    data = data.iloc[:, 2:6]
    # 2：实例化一个转换器类
    transfer = StandardScaler()
    # 3：调用fit_transform
    data_new = transfer.fit_transform(data)
    print("标准化data_new:\n", data_new)
    return None
if __name__=="__main__":
    # 归一化结果
    # minmax_dema()
    # 归一化标准化
    stand_demo()
