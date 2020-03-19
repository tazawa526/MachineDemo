#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   sklearn_004.py    
@Contact :   tianze05@163.com
@License :   (C)Copyright 2020-2030, LT.CT

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2020/3/19 14:21   tazawa      1.0         KNN
'''
"""
sklearn.neighbors.KNeighborsClassifier(n_neighbors=5,algorithm='auto')
n_neighbors:int (可选) ，k_neighbors 查询默认使用的邻居数
algorithm：{auto,ball_tree,kd_tree,brute},可选用于计算邻居的算法
   ball_tree 将会使用BallTree,
   kd_tree 将使用KDTree
   auto 将尝试根据传递给fit方法的值来决定最合适的算法
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
def knn_demo():
    """
    用KNN算法对鸢尾花进行分类
    :return:
    """
    # 1、获取鸢尾花数据集
    iris = load_iris()
    # 2.划分数据集
    x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=6)
    # 3.特征工程---标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)  #训练值标准值
    x_test = transfer.transform(x_test)   #测试值标准值
    # 4.KNN算法预评估
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train,y_train)
    # 5.模型评估
    # 方法1：直接比对真实值和预测值
    y_predict = estimator.predict(x_test)
    print("y_predict:\n",y_predict)
    print("直接比对真实值和预测值:\n",y_test==y_predict)
    # 方法2：计算准确率
    score = estimator.score(x_test,y_test)
    print("准确率为:\n",score)
    return  None
if __name__=="__main__":
    # KNN算法对鸢尾花的预测
    knn_demo()