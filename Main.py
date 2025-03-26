import numpy as np
from data_loader import carregar_dados
from preprocess import preprocessar_dados, criar_dataset
from modelo import criar_modelo
from train import treinar_modelo
from predict import fazer_previsao, plotar_previsoes

# Carregar e processar os dados
df = carregar_dados()
dados_normalizados, escala = preprocessar_dados(df)

# Criar dataset de sequências temporais
time_step = 60
X, y = criar_dataset(dados_normalizados, time_step)

# Reshape para formato LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Divisão treino/teste
limite_treino = int(len(X) * 0.8)
X_treino, X_teste = X[:limite_treino], X[limite_treino:]
y_treino, y_teste = y[:limite_treino], y[limite_treino:]

# Criar e treinar o modelo
modelo = criar_modelo(time_step)
modelo = treinar_modelo(modelo, X_treino, y_treino)

# Fazer previsões
previsoes = fazer_previsao(modelo, X_teste, escala)

# Plotar previsões vs valores reais
plotar_previsoes(y_teste, previsoes)

