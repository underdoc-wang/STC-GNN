from keras.models import Sequential
from keras.layers import InputLayer, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization


# global
H, W, C = (20, 10, 6)     # image size


def get_ConvLSTM(timestep, n_layers, n_units):

    model = Sequential()
    model.add(InputLayer(input_shape=(timestep, H, W, C)))

    for i in range(n_layers - 1):
        model.add(ConvLSTM2D(filters=n_units, kernel_size=(3, 3),
                             padding='same', return_sequences=True))
        model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=n_units, kernel_size=(3, 3),
                         padding='same', return_sequences=False))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=C, kernel_size=(1, 1), padding='same', activation='sigmoid'))

    return model