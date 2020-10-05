import numpy as np
from timeseries import generate_time_series
import data

x, y = generate_time_series('WIKI/MSFT', 'Close', 24, 10, debug_mode=True)

for t in data.DOW:
    if t != 'WIKI/MSFT':
        x, y = generate_time_series(t, 'Close', 24, 10, x=x, y=y, debug_mode=True)

np.savetxt('datas/x_nasdaq_close_dow24.txt', x)
np.savetxt('datas/y_nasdaq_close_dow24.txt', y)
