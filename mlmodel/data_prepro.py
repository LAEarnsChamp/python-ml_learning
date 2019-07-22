#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np

def normalization(datalist, place=2):
    """data normalizaton

    datalist:data list
    palce:decimal places, default is 2
    return:normalized data generator
    """
    mindata = min(datalist)
    maxdata = max(datalist)

    generator = (round((x - mindata)/(maxdata-mindata), place) for x in datalist)
    return generator

def standardization(datalist, place=2):
    """data z-score normalizaton

    datalist:data list
    palce:decimal places, default is 2
    return:z-score normalized data generator
    """
    mean = np.mean(datalist)
    std = np.std(datalist)   

    generator = (round((x - mean)/std, place) for x in datalist)
    return generator