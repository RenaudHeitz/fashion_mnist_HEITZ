from keras.datasets import fashion_mnist
from keras.layers import Dense, MaxPool2D, Conv2D, Dropout
from keras.layers import Flatten, InputLayer
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
from keras.initializers import Constant
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
clf.add(
    Conv2D(
        64, (3, 3), 
        padding='same', 
        bias_initializer=Constant(0.01), 
        kernel_initializer='random_uniform'
    )
)
clf.add(MaxPool2D(padding='same'))

clf.add(
    Conv2D(
        32, (3, 3), 
        padding='same', 
        bias_initializer=Constant(0.01), 
        kernel_initializer='random_uniform', 
        input_shape=(1, 28, 28)
    )
)
clf.add(MaxPool2D(padding='same'))

clf.add(Flatten())

clf.add(
    Dense(
        128,
        activation='relu',
        bias_initializer=Constant(0.01), 
        kernel_initializer='random_uniform',         
    )
)

clf.add(Dense(10, activation='softmax'))

clf.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy']
)

print(clf.summary())

clf.fit(
    x_train, 
    y_train, 
    epochs=20, 
    batch_size=32, 
    validation_data=(x_test, y_test)
)

clf.evaluate(x_test, y_test)

