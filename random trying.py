import numpy as np
from tensorflow.keras import layers, models, optimizers, datasets, utils, Sequential

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

x_train = utils.normalize(x_train.reshape(60000, 28*28))
x_test = utils.normalize(x_test.reshape(10000, 28*28))

model = Sequential()

model.add(layers.Dense(128, activation='relu', input_shape=(28*28,)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

optimizer = optimizers.Adam(lr=0.01)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_test, y_test))
