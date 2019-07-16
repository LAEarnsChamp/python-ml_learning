#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import mlmodel

data = np.array([
    [154.1, 1.1],
    [126, 2.2],
    [70, 2.4],
    [196, 2.5],
    [161, 2.7],
    [371, 4.4],
    ])

feat = data[:, 0]
label = data[:, -1]
a = mlmodel.models.knn(2, 500.111, feat, label)
print(a)