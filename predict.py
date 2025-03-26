import matplotlib.pyplot as plt

def fazer_previsao(modelo, X_teste, escala):
    presivoes = modelo.predict(X_teste)
    previsoes = escala.inverse_transform(previsoes)
    return previsoes

def plotar_previsoes(Y_teste, previsoes):
    plt.figure(figsize=(10, 5))
    plt.plot(Y_teste, label='Valores Reais', color='Blue')
    plt.plot(previsoes, label='Previsões', color='Red')
    plt.title('Previsões de Preço de Bitcoin')
    plt.xlabel('amostras')
    plt.ylabel('Preço')
    plt.legend()
    plt.show()
