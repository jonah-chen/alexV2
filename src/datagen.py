import numpy as np
from timeseries import generate_time_series
import data

x, y = generate_time_series('WIKI/MSFT', 'Open', 8, debug_mode=True)

for t in data.NASDAQ:
    if t != 'WIKI/MSFT':
        x, y = generate_time_series(t, 'Open', 8, x=x, y=y, debug_mode=True)

np.savetxt('datas/x_nasdaq_open_timeseries8.txt', x)
np.savetxt('datas/y_nasdaq_open_timeseries8.txt', y)
