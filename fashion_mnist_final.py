from __future__ import print_function

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np

batch_size = 256
num_classes = 10
epochs = 18

img_rows, img_cols = 28, 28

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
	x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	input_shape = (1, img_rows, img_cols)
else:
	x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
	x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
	input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation = 'relu', input_shape = input_shape))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

# Save the checkpoint in the /output folder
filepath = "./best.hdf5"

if os.path.exists(filepath) == True:
	model.load_weights(filepath)

# Keep only a single checkpoint, the best over test accuracy.
checkpoint = ModelCheckpoint(filepath,
                            monitor='val_acc',
                            verbose=1,
                            save_best_only=True,
                            mode='max')

reduce_lr = ReduceLROnPlateau(monitor='acc', factor=0.2,
                              patience=2, min_lr=0.01, min_delta=0.01)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

print(model.summary())
model.fit(x_train, y_train, batch_size=batch_size, shuffle=True ,epochs=5, verbose=1, validation_data=(x_test, y_test), callbacks=[checkpoint, reduce_lr])

#Define data aumentation
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1)

# fit the data augmentation
datagen.fit(x_train)

# setup generator
model.fit_generator(datagen.flow(x_train, y_train, batch_size=250),
                              steps_per_epoch=int(np.ceil(x_train.shape[0] / float(250))),
                              epochs=10,
                              validation_data=(x_test, y_test))

model.fit(x_train, y_train, batch_size=batch_size, shuffle=True ,epochs=15, verbose=1, validation_data=(x_test, y_test), callbacks=[checkpoint, reduce_lr])

score = model.evaluate(x_test, y_test, verbose = 0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])