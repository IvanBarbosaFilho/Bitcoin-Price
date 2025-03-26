from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def treinar_modelo(modelo, X_treino, Y_trein, epochs=100, batch_size=32):
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, min_delta=0.0001)

    historico = modelo.fit(
            X_treino, Y_treino,
            validation_split=0.2,
            epochs=epochs,
            batch_size_=batch_size,
            callbacks=[early_stop],
            verbose=1
)

    plt.figure(figsize=(10,5))
    plt.plot(historico.history['loss'], label='Treino')
    plt.plot(historico.history['val_loss'], label='Validação')
    plt.title('Histórico de perdas do modelo')
    plt.Xlabel('Epochs')
    plt.Ylabel('Perda')
    plt.lengend()
    plt.show()

    return modelo
