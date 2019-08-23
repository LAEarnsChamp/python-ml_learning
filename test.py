#!/usr/bin/python
# -*- coding: UTF-8 -*-
# test

import numpy as np
from mlmodel import models
from numpy.random import random_sample

# # fig = plt.figure()
# N = 3
# # 设椭圆中心center
# cx = 5
# cy = 6
# a = 1/8.0
# b = 4
# X, scale = 2*a*random_sample((N,))+cx-a, 60
# Y = [2*b*np.sqrt(1.0-((xi-cx)/a)**2)*random_sample()+cy-b*np.sqrt(1.0-((xi-cx)/a)**2) for xi in X]
# X1, scale = 2*a*random_sample((N,))+cx-a, 60
# Y1 = [2*b*np.sqrt(1.0-((xi-cx)/a)**2)*random_sample()+cy-b*np.sqrt(1.0-((xi-cx)/a)**2) for xi in X1]

D1 = np.array([
    [1, 1, 1],
    [4, 4, 4]
    ])
D2 = np.array([
    [2, 12, 2],
    [4, 5, 6]
    ])

a, b, c = models.lda_bi_classification(D1, D2)
print(a)
print(b)
print(c)
# print(D1, D2)
# m1 = np.mean(D1, axis=1)
# m1 = m1.reshape((1, len(D1)))
# print("m1:\n", m1)
# m2 = np.mean(D2, axis=1)
# m2 = m2[None, ]
# print("m2:\n", m2)
# SB = np.dot((m1-m2).T, (m1-m2))
# S1 = np.dot(D1-m1.T, (D1-m1.T).T)
# print("s1:\n", S1)
# S2 = np.dot(D2-m2.T, (D2-m2.T).T)
# SW = S1+S2
# print("SB:\n", SB)
# print("SW:\n", SW)
# S = np.dot(np.linalg.inv(SW), SB)
# evalue, evec = np.linalg.eig(S)
# w = np.linalg.inv(SW) * (m1 - m2)
# print("evalue:\n", evalue)
# print("evec:\n", evec)
# print("w:\n", w)

# # data = np.array([
# #     [-1, 1, 3],
# #     [-2, 1, 5],
# #     [-0, 1, 3],
#     [-2, 1, 3],
#     [-3, 1, 5],
#     [-4, 1, 7]
#     ])
# print(models.linear_model(data))

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


# # if __name__ == '__main__':
# #     train()
# =======

# import numpy as np
# # import mlmodel
# # import sys

# # # data = np.array([
# # #     [154.1, 1.1],
# # #     [126, 2.2],
# # #     [70, 2.4],
# # #     [196, 2.5],
# # #     [161, 2.7],
# # #     [371, 4.4],
# # #     ])

# # # feat = data[:, 0]
# # # label = data[:, -1]
# # # a = mlmodel.models.knn(2, 500.111, feat, label)
# # # print(a)

# # list1 = [0.4, 0.6, 1]
# # print(list(mlmodel.data_prepro.standardization(list1)))

# data = np.array([
#     [80, 200],
#     [95, 230],
#     [104, 245],
#     [112,274],
#     [125, 259],
#     [135, 262]
#     ])

# feature = data[:, 0:1]
# label = np.expand_dims(data[:, -1], axis=1)

# k = 1
# m = 1
# weight = np.array([
#     [k],
#     [m]
#     ])
# # 拼接
# feature_matrix = np.append(feature, np.ones(shape=(6, 1)), axis=1)
# # 乘
# dmatrix = np.dot(feature_matrix, weight) - label
# # 转置
# np.dot(feature_matrix.T, dmatrix) * 2 /len(data[:, 0])

# print(feature.shape)
# >>>>>>> 8b6f9b59d57657a10fd733b4114f7df3f2fa9f20
