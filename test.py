#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
# import mlmodel
# import sys

# # data = np.array([
# #     [154.1, 1.1],
# #     [126, 2.2],
# #     [70, 2.4],
# #     [196, 2.5],
# #     [161, 2.7],
# #     [371, 4.4],
# #     ])

# # feat = data[:, 0]
# # label = data[:, -1]
# # a = mlmodel.models.knn(2, 500.111, feat, label)
# # print(a)

# list1 = [0.4, 0.6, 1]
# print(list(mlmodel.data_prepro.standardization(list1)))

data = np.array([
    [80, 200],
    [95, 230],
    [104, 245],
    [112,274],
    [125, 259],
    [135, 262]
    ])

feature = data[:, 0:1]
label = np.expand_dims(data[:, -1], axis=1)

k = 1
m = 1
weight = np.array([
    [k],
    [m]
    ])
# 拼接
feature_matrix = np.append(feature, np.ones(shape=(6, 1)), axis=1)
# 乘
dmatrix = np.dot(feature_matrix, weight) - label
# 转置
np.dot(feature_matrix.T, dmatrix) * 2 /len(data[:, 0])

print(feature.shape)