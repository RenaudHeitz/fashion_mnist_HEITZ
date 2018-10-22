# https://medium.com/@lukaszlipinski/fashion-mnist-with-keras-in-5-minuts-20ab9eb7b905

from keras.datasets import fashion_mnist
from keras.layers import Dense, MaxPool2D, Conv2D, Dropout, LeakyReLU
from keras.layers import Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from keras.initializers import Constant
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


# Load data
# Function load_minst is available in git.
(x_train, y_train), (x_test, y_test)  = fashion_mnist.load_data()


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (x_train.shape[0],1,28,28))
x_test = np.reshape(x_test, (x_test.shape[0],1,28,28))
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)


clf = Sequential()

clf.add(
    InputLayer(input_shape=(1, 28, 28))
)
clf.add(
    BatchNormalization()
)

clf.add(Conv2D(64, (4, 4), padding='same'))
clf.add(LeakyReLU())
clf.add(MaxPool2D(padding='same'))


clf.add(
    Conv2D(
        128, (4, 4), 
        padding='same', 
        bias_initializer=Constant(0.01), 
        kernel_initializer='random_uniform',
    )
)

clf.add(
    Conv2D(
        64, (4, 4), 
        padding='same', 
        bias_initializer=Constant(0.01), 
        kernel_initializer='random_uniform',
    )
)
clf.add(LeakyReLU())

clf.add(MaxPool2D(padding='same'))

clf.add(Flatten())

clf.add(
    Dense(
        256,
        activation='relu',
        bias_initializer=Constant(0.01), 
        kernel_initializer='random_uniform',         
    )
)
clf.add(Dropout(0.5))

clf.add(Dense(64, activation='relu'))

clf.add(
    BatchNormalization()
)

clf.add(Dense(10, activation='softmax'))

clf.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

print(clf.summary())



datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

clf.fit_generator(datagen.flow(x_train, y_train, batch_size=250),
                              steps_per_epoch=int(np.ceil(x_train.shape[0] / float(250))),
                              epochs=5,
                              validation_data=(x_test, y_test))

checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)

clf.fit(
    x_train, 
    y_train,
    shuffle=True, 
    epochs=10, 
    batch_size=250, 
    validation_data=(x_test, y_test),
    callbacks=checkpointer
)

clf.evaluate(x_test, y_test)

