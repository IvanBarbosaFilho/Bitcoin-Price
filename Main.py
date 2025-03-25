#Bibliotecas 

import numpy as np
import datetime as dt
import pandas as pd
import ccxt
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import Earlystopping

#Conexão com a Binance

binance = ccxt.binance()
symbol = 'BTC/USDT'
timeframe = '1m'
ohlcv = binance.fetch_ohlcv(symbol, timeframe, limit=1000)

#Preparando os Dados e convertendo pra DataFrame

df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 
                                  'low', 'close', 'volume'])

df['timestamp'] = pd.to_datetime(df['timestamp'], unit='me')
df.set_index('timestamp', inplace=True)

dados = df[['close']].values

#Transformando para ficar mais legiveis pro modelo

escala = MinMaxScaler(feature_range=(0, 1))
dados_normalizados = escala.fit_transform(dados)

#Dando entrada e saida para o modelo (X e Y)

def criar_dataset(dataset, time_step=60):
    
    X, Y = [], []

    for 1 in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]

        X.append(a)

        Y.append(dataset(i + time_step, 0))

    return np.array(X), np.array(Y)    

#Sequencias

time_step = 60
X,Y = criar_dataset(dataset, time_step)

#Reformando elas para um formato que LSTM entenda

X = X.reshape(X.shape[0], X.shape[1], 1)

#Adicionando limite de treino

Limite_Treino = int(len(X) * 0.8)

X_treino, X_teste = X[:Limite_Treino], X[Limite_Treino]
Y_treino, Y_teste = Y[:Limite_Treion], Y[Limite_Treino]

#Modelo LSTM utilizando Dropout

modelo = Sequential([
    LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    LSTM(50, return_sequence=False),
    Dropout(0.2),
    Dense(25, activision='relu'),
    Dropout(0.2),
    Dense(1)
])

#Compilação do modelo

modelo.compile(optimizer='adam', loss='mean_squared_error')

#Early Stooping

early_stop = Earlystopping(
        monitor='vel_loss',
        patience=10,
        restore_best_weight=True,
        min_delta=0.0001
)

#treinamento de modelo

historico = modelo.fit(
        X_treino, Y_treino,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
)

#Histórico de perda do modelo

plt.figure(figsize=(10,5))
plt.plot(historico.history['loss'], label='Treino')
plt.plot(historico.history['val_loss'], label='Validação')
plt.title('Histórico de perda do modelo')
plt.Xlabel('Epochs')
plt.Ylabel('Perda')
plt.legend()
plt.show()

#Previsão

def fazer_previsao(modelo, dados_entrada, escala):
    previsao = modelo.predict(dados_entrada)
    previsao = escala.inverse_transform(previsao)

    return previsao

previsões = fazer_previsao(modelo, X_teste, escala)

plt.figure(figsize=(10,5))
plt.plot(Y_teste, label='Valores Reais', color='Blue')
plt.plot(previsoes, label='Previsões', color='Red')
plt.title('Previsões do preço de Bitcoin')
plt.Xlabel('Amostras')
plt.Ylabel('Preço')
plt.legend()
plt.show()
