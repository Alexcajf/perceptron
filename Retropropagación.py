import tensorflow as tf
import numpy as np

# Crear un conjunto de datos de ejemplo
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [1], [1], [0]], dtype=np.float32)

# Definir el modelo de red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(2, activation='sigmoid'),  # Capa oculta con función de activación sigmoide
    tf.keras.layers.Dense(1, activation='sigmoid')   # Capa de salida con función de activación sigmoide
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mse')  # Usar MSE (error cuadrático medio) como función de pérdida

# Entrenar el modelo usando retropropagación
model.fit(X, y, epochs=1000, verbose=0)  # Realizar un gran número de épocas para entrenamiento

# Hacer predicciones
predictions = model.predict(X)

# Imprimir las predicciones
print("Predicciones:")
print(predictions)
