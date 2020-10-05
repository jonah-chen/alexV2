import numpy as np
import quandl
import pandas as pd
from sklearn import preprocessing
from collections import deque

quandl.ApiConfig.api_key = 'MdTYhiXQR_LnUHzq-qma'


def generate_x(ts, data, length, begin):
    norm = ts[data][begin]
    x = np.array([ts[data][begin:begin + length].tolist()]) / norm
    y = ts[data][begin + length] / norm
    return x, y, norm


def generate_time_series(tinker, data, length, begin=0, x=None, y=None, norm_array=None,
                         debug_mode=False, normalization_coefficients=False):
    comp = quandl.get(tinker)
    o = 0
    k = 0
    if x is None:
        x, y, norm_array = generate_x(comp, data, length, begin)
        o = 1
    for i in range(begin + o, len(comp) - length - 1):
        norm = comp[data][i]
        x = np.concatenate((x, np.array([comp[data][i:i + length].tolist()]) / norm), axis=0)
        y = np.append(y, comp[data][i + length + 1] / norm)
        if normalization_coefficients:
            norm_array = np.append(norm_array, norm)
        if debug_mode:
            k += 1
            if k % 1000 == 0:
                print(tinker + ': ' + str(k) + ' datas')
        if normalization_coefficients:
            return x, y, norm_array
    return x, y
