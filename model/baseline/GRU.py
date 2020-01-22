from keras.models import Sequential
from keras.layers import InputLayer, GRU, LSTM, Dense, Reshape


# global
H, W, C = (20, 10, 6)     # image size


def get_GRU(timestep, n_layers, n_units):
    model = Sequential()

    model.add(InputLayer(input_shape=(timestep, H, W, C)))
    model.add(Reshape(target_shape=(timestep, H*W*C)))     # reshape to fit GRU

    for i in range(n_layers - 1):
        model.add(LSTM(units=n_units, return_sequences=True))
    model.add(LSTM(units=n_units))

    model.add(Dense(units=H*W*C, activation='sigmoid'))
    model.add(Reshape(target_shape=(H, W, C)))      # reshape back to image


    return model