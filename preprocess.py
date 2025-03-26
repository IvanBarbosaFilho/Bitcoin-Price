import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocessar_dados(df):
    escala = MinMaxScaler(feature_range=(0, 1))
    dados_normalizados = escala.fit_transform(df[['close']].values)
    return dados_normalizados, escala

def criar_dataset(dataset, time_step=60):
    X,Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)
