from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

def criar_modelo(time_step):
    modelo = Sequential([
        LSTM(50,return_sequences=True, input_shape=(time_step, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        LSTM(25, activision='relu'),
        Dropout(0.2),
        Dense(1)
])
    modelo.compile(optimizer='adam',loss='mean_squared_error')
    return modelo
