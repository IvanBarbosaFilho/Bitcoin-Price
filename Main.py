#Bibliotecas 

import numpy as np
import datetime as dt
import pandas as pd
import ccxt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

#Conexão com a Binance

binance = ccxt.binance()

symbol = 'BTC/USDT'
timeframe = '1m'

ohlcv = binance.fetch_ohlcv(symbol, timeframe, limit=1000)

ohlcv_np = np.array(ohlcv)

#Preparando os Dados

escala = MinMaxScaler(feature_range=(0, 1))
ohlcv_escala = scaler.fit_transform(ohlcv_np[:, 1:])


