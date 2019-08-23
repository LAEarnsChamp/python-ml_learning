#!/usr/bin/python
# -*- coding: UTF-8 -*-
# ml models

import numpy as np
from collections import Counter


def knn(k, pred_point, feature, label):
    '''
    knn

    k: k
    pred_point: point to predicted
    feature: training features, list format
    label: training labels, list format
    '''
    distance = list(map(lambda x: abs(pred_point-x), feature))
    sortindex = np.argsort(distance)
    sortlabel = label[sortindex]

    return Counter(sortlabel[0: k]).most_common(1)[0][0]


def linear_model(data, learning_rate=0.001, loop=10000):
    """
    y = X.T * w + b

    data: np matrix contained features, and labels in the LAST column
    learning_rate: controlling the convergence speed
    loop: controlling the convergence speed

    return: weights matrix
    """
    m = len(data)
    feature = np.append(data[:, 0:-1], np.ones(shape=(m, 1)), axis=1)
    weights = np.ones(shape=(feature.shape[-1], 1))
    label = data[:, -1:]

    for x in range(0, loop):
        mse = 2/m * np.dot(feature.T, (np.dot(feature, weights) - label))
        weights -= learning_rate * mse

    return weights


def lda_bi_classification(class1, class2):
    """
    lda二分类模型
    sklearn.lda is recommended

    class1: data from the first class,
    each row represents all the data of one dimension
    class1: data from the second class,
    each row represents all the data of one dimension

    return: projection hyper-plane, eigenvalue, eigenvector
    """
    mean1 = np.mean(class1, axis=1)
    mean1 = mean1.reshape((1, len(class1)))
    mean2 = np.mean(class2, axis=1)
    mean2 = mean2.reshape((1, len(class1)))
    SB = np.dot((mean1-mean2).T, (mean1-mean2))
    S1 = np.dot(class1-mean1.T, (class1-mean1.T).T)
    S2 = np.dot(class2-mean2.T, (class2-mean2.T).T)
    SW = S1+S2
    S = np.dot(np.linalg.inv(SW), SB)
    evalue, evector = np.linalg.eig(S)
    w = np.linalg.inv(SW) * (mean1 - mean2)

    return w, evalue, evector
