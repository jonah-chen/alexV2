import quandl
import numpy as np
import pandas as pd
from sklearn.preprocessing.data import StandardScaler
from collections import deque

quandl.ApiConfig.api_key = 'MdTYhiXQR_LnUHzq-qma'


def generate_time_series(tinker, data, length, begin=0, debug_mode=False):
    if debug_mode:
        print(tinker)
        print('---------------------------------------------------------')
    comp = quandl.get(tinker)[data]

    x = []
    y = []

    prev_days = deque(maxlen=length)
    for i in range(begin, len(comp) - 1):
        prev_days.append(comp[i])
        if len(prev_days) == length:
            x.append(np.array(prev_days).flatten())
            y.append(comp[i + 1])

    return np.array(x), np.array(y)


def generate_norm_time_series(tinker, data, length, begin=0, debug_mode=False):
    if debug_mode:
        print(tinker)
    comp = quandl.get(tinker)[data]
    comp = comp.reshape(-1, 1)
    sc = StandardScaler()
    sc.fit(comp)
    comp = sc.transform(comp)

    x = []
    y = []

    prev_days = deque(maxlen=length)
    for i in range(begin, len(comp) - 1):
        prev_days.append(comp[i])
        if len(prev_days) == length:
            x.append(np.array(prev_days).flatten())
            y.append(comp[i + 1][0])

    return np.array(x), np.array(y)


# shuffle the data later
def generate_norm_pct_time_series(tinker, data, length, begin=0, debug_mode=False, x_init=[], y_init=[], x_2_init=[],
                                  y_2_init=[]):
    if debug_mode:
        print(tinker)
    # acquire data as % change
    prices = quandl.get(tinker)[data]
    comp = prices.pct_change()
    comp.dropna(inplace=True)
    comp = np.reshape(comp.to_numpy(), (-1, 1))
    # normalize data
    sc = StandardScaler()
    sc.fit(comp)
    comp = sc.transform(comp)
    # format normalized data

    x, y, x_2, y_2 = x_init, y_init, x_2_init, y_2_init
    
    x_b = comp[begin:begin+length].tolist()
    x.append(x_b)
    y.append(comp[begin+length][0])
    x_2.append(prices[begin+length-1])
    y_2.append(prices[begin + length])
    for i in range(begin + length, len(comp)-1):
        x_b = x_b[1:]
        x_b.append(comp[i].tolist())
        x.append(x_b)
        y.append(comp[i+1][0])
        x_2.append(prices[i])
        y_2.append(prices[i+1])

    return x, y, x_2, y_2


def denormalize(x, tinker, data, begin=0, x_2=None):
    prices = quandl.get(tinker)[data]
    comp = prices.pct_change()
    comp.dropna(inplace=True)
    comp = np.reshape(comp.to_numpy(), (-1, 1))
    sc = StandardScaler()
    sc.fit(comp)
    if x_2 is not None:
        return (1 + sc.inverse_transform(x)) * x_2
    return sc.inverse_transform(x)

# w = generate_time_series('WIKI/MSFT', ['Close'], 5, begin=5, debug_mode=True)
#
# z = generate_time_series('WIKI/AAPL', ['Close'], 5, begin=5, debug_mode=True)
#
# x = np.append(w,z,axis=0)
#
# print(w.shape)
# print(z.shape)
# print(x.shape)
