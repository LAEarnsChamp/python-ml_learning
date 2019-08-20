#!/usr/bin/python
# -*- coding: UTF-8 -*-
# test

import numpy as np
from mlmodel import models

data = np.array([
    [-1, 1, 3],
    [-2, 1, 5],
    [-0, 1, 3],
    [-2, 1, 3],
    [-3, 1, 5],
    [-4, 1, 7]
    ])
print(models.linear_model(data))

# import matplotlib.pyplot as plt
# import math

# learning_rate = 0.001

# data = np.array([
#     [1, 1, 5],
#     [2, 2, 9],
#     [0, 2, 5],
#     [2, 0, 5],
#     [3, 1, 9],
#     [4, 2, 13]
#     ])

# m = len(data)
# feature = np.append(data[:, 0:-1], np.ones(shape=(m, 1)), axis=1)
# weights = np.ones(shape=(feature.shape[-1], 1))
# label = data[:, -1:]
# print("learning_rate:\n", learning_rate)
# print("m\n", m)
# print("feature:\n", feature)
# print("weights:\n", weights)
# print("label:\n", label)
# print("data.T:\n", data.T)


# for x in range(0, 1000000):
#     mse = 2/m * np.dot(feature.T, (np.dot(feature, weights) - label))
#     weights -= learning_rate * mse
# print(weights)

# feature = data[:, 0:1]
# label = data[:, -1:]

# feature_matrix = np.append(feature, np.ones(shape=(6, 1)), axis=1)


# def gradent_decent():
#     predict = 1 / (1+np.exp(-(np.dot(feature_matrix, weights))))
#     return np.dot(feature_matrix.T, predict-label)


# def train():
#     for x in range(0, 1000000):
#         slop = gradent_decent()
#         global weights
#         weights -= learning_rate * slop
#     print(weights)


# if __name__ == '__main__':
#     train()
# n = 100000
# print(math.pow((1+1.0/n), n))


# points = np.array([
#     [1, 2],
#     [3, 4],
#     [5, 6],
#     ])

# array = np.array([2,0])
# newpoints = points + array

# plt.plot(points[:, 0], points[:, -1]) 
# plt.plot(newpoints[:, 0], newpoints[:, -1])
# plt.show()

# data = np.array([
#     [80, 200],
#     [95, 230],
#     [104, 245],
#     [112, 274],
#     [125, 259],
#     [135, 262],
#     ])

# feature = data[0:2, 0]
# label = data[:, -1: -2]

# print(feature, label)

# A = np.array([
#     [3, 1, 2],
#     [-5, 4, 1],
#     [0, 3, -8],
#     ])

# B = np.array([
#     [0, -5, 1],
#     [3, 2, -1],
#     [10, 0.5, 4],
#     ])

# print(np.dot(A, B))

# data = np.array([
#     [80, 200],
#     [90, 230],
#     [104, 245],
#     [112, 247],
#     [125, 259],
#     [135, 262],
#     ])

# k = 1
# m = 1

# xarray = data[:, 0]
# yreal = data[:, -1]
# learning_rate = 0.00001


# def grandentdecent():
#     # 构建损失函数对常数m的偏导数
#     mslop = 0
#     for index, x in enumerate(xarray):
#         mslop += k * x + m - yreal[index]
#     mslop = mslop * (2/len(xarray))
#     # print("对m求导：%f"%mslop)

#     # 构建损失函数对斜率k的偏导数
#     kslop = 0
#     for index, x in enumerate(xarray):
#         kslop += (k * x + m - yreal[index]) * x
#     kslop = kslop * (2/len(xarray))
#     print("对k求导：%f x:%f m:%f"%(kslop, k, m))

#     return(mslop, kslop)


# def train():
#     for i in range(0, 100000):
#         mslop, kslop = grandentdecent()
#         global m
#         m -= mslop * learning_rate
#         global k
#         k -= kslop * learning_rate
#         if (abs(mslop) < 0.5) and (abs(kslop) < 0.5):
#             break
#     print("k:{} m:{}".format(k, m))


# if __name__ == '__main__':
#     train()