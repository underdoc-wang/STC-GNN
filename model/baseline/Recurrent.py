from keras.models import Sequential
from keras.layers import InputLayer, GRU, ConvLSTM2D, BatchNormalization


# global
H, W, C = (20, 10, 6)     # image size


def get_GRU(timestep, n_layers, n_units):
    model = Sequential()
    model.add(InputLayer(input_shape=(timestep, C)))

    for i in range(n_layers - 1):
        model.add(GRU(units=n_units, return_sequences=True))
    model.add(GRU(units=C, activation='sigmoid'))
    return model


def get_ConvLSTM(timestep, n_layers, n_units):
    model = Sequential()
    model.add(InputLayer(input_shape=(timestep, H, W, C)))

    for i in range(n_layers - 1):
        model.add(ConvLSTM2D(filters=n_units, kernel_size=(3, 3),
                             padding='same', return_sequences=True))
        model.add(BatchNormalization())
    model.add(ConvLSTM2D(filters=C, kernel_size=(3, 3), activation='sigmoid',
                         padding='same', return_sequences=False))
    return model
