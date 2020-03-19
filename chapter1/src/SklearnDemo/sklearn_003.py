#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   sklearn_003.py    
@Contact :   tianze05@163.com
@License :   (C)Copyright 2020-2030, LT.CT

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/3/13 0:22   tazawa      1.0         特征降维
'''
# 特征选择
"""
Filter(过滤式)：主要探究特征本身特点、特征与特征和目标值之间关联
    方差选择法：低方差特征过滤
    相关系数:特征与特征之间的相关程度 
        特征与特征之间相关性很高
         1）选择其中一个  2）加权求和  3）主成分分析
 
  低方差特征过滤： 删除低方差的一些特征，再结合方差的大小来考虑这个方式的角度
     特征方差小：某个特征大多样本的值比较相近
     特征方差大：某个特征很多样本的值都有差别
  sklearn.feature_selection.VarianceThreshold(threshold=0.0)
    删除所有低方差特征
    Variance.fit_transform(X)
    X:numpy array格式的数据[n_samples,n_features]
    返回值：训练集差异低于threshod的特征将被删除，默认值是保留所有非零方差特征，既删除所有样本中具有相同值的特征
    
Embedded(嵌入式)：算法自动 选择特征(特征与目标值之间的关联)
  决策树：信息熵、信息增益
  正则化：L1，L2
  深度学习：卷积等
"""
"""
主成分分析(PCA)
定义：高维数据转化为低维数据的过程，在此过程中可能会舍弃原有数据、创造新的变量
作用：是数据维数压缩，尽可能降低原数据的维数(复杂度)，损失少量信息。
应用：回归分析或者聚类分析当中

sklearn.decomposition.PCA(n_components=None)
将数据分解为较低维数空间
n_components:
   小数：表示保留百分之多少的信息
   整数：减少到多少特征
   PCA.fit_transform(X) X:numpy array格式的数据
   [n_samples,n_features]
   返回值：转换后指定维度的array
"""

from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
def variance_demo():
    """
    过滤低方差特征
    :return:
    """
    # 1：获取数据
    filepath = r'E:\Python\txt\GP\HB\gpmx.xls'
    data = pd.read_excel(filepath, encoding='utf-8', sheet_name=0)
    data = data.iloc[:, 2:6]
    # 2：实例化一个转换器类
    transfer = VarianceThreshold(threshold=2)
    # 3：调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new,type(data_new),data_new.shape)

    # 计算某两个变量之间的相关系数
    r = pearsonr(data['开盘'],data['高'])
    print("相关系数：\n",r)
    # 可视化
    plt.figure(figsize=(20,8),dpi=100)
    plt.scatter(data['开盘'],data['高'])
    plt.show()
    return None

def pca_demo():
    """
    PCA 降维
    :return:
    """
    data =[[2,8,4,5],[6,3,0,8],[5,4,9,1]]
    # 1、实例化一个转换器类
    transfer = PCA(n_components=2)
    # 2、调用fit_transform
    data_new = transfer.fit_transform(data)
    print("主分析降维:\n",data_new)
if __name__=="__main__":
    # PCA降维
    pca_demo()
    # variance_demo()