import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization
import time
import quandl
import matplotlib.pyplot as plt
from timeseries import generate_time_series
from sklearn.utils import shuffle
from sklearn import preprocessing

quandl.ApiConfig.api_key = 'MdTYhiXQR_LnUHzq-qma'

NAME = f"dowLSTM2x256+dropout0.4+32{int(time.time())}"

x = np.genfromtxt('datas/x_dow_close_timeseries24.txt')
y = np.genfromtxt('datas/y_dow_close_timeseries24.txt')


x = x.reshape(x.shape[0], x.shape[1], 1)
tensorboard = TensorBoard(log_dir=f'logs/{NAME}')
model = Sequential()
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(LSTM(128))
model.add(Dropout(0.4))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='relu'))

opt = tf.keras.optimizers.Adam(lr=0.0005, decay=5e-7)

model.compile(optimizer=opt,
              loss='mse',
              metrics=['mean_absolute_percentage_error'])

model.fit(x, y, batch_size=32, epochs=30, callbacks=[tensorboard], validation_split=0.2)
