# -*- coding:utf-8 -*-

"""
@ide: PyCharm
@author: mesie
@date: 2021/7/2 10:25
@summary:
"""
from tensorflow.python.keras import backend as K

def sampledsoftmaxloss(y_true, y_pred):

    return K.mean(y_pred)