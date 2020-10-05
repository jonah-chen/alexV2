import pandas as pd
import numpy as np
import time
import quandl
from tensorflow.keras import layers, models, optimizers, metrics
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from timeseries import generate_time_series

quandl.ApiConfig.api_key = 'MdTYhiXQR_LnUHzq-qma'

x = np.genfromtxt('datas/x_dow_close_timeseries24.txt')
y = np.genfromtxt('datas/y_dow_close_timeseries24.txt')

x, y = shuffle(x, y)

NAME = f"MultiStockFeedfoward_32-96-32{int(time.time())}"

tensorboard = TensorBoard(log_dir=f'logs/{NAME}')
model = models.Sequential()
model.add(layers.Dense(32, activation='relu', input_shape=(24,)))
model.add(layers.Dense(96, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1, activation='relu'))

optimizer = optimizers.Adam(lr=2e-3, decay=4e-7)
model.compile(optimizer=optimizer,
              loss='mse',
              metrics=['mean_absolute_percentage_error'])

model.summary()

model.fit(x, y, batch_size=32, epochs=30, validation_split=0.2,
          callbacks=[tensorboard])
