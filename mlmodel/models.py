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
