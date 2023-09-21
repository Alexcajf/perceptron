import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Definiendo la funcion f(x1, x2)
def custom_function(x):
    return 10 - tf.exp(-x[:, 0] ** 2 + 3 * x[:, 1] ** 2)

# Crenado el dataset con valores aleatorios
X = np.random.rand(100, 2)  # 100 data points con 2 características
y = custom_function(X)

# Definiendo un perceptrón multicapa
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)  # Capa de salida con una neurona
])

# Compilando el modelo
model.compile(optimizer='adam', loss='mse')  # Usando "mean squared error loss"

# Entrenando el modelo
history = model.fit(X, y, epochs=100, verbose=2)

# Graficando la función custom_function
x1 = np.linspace(0, 1, 100)
x2 = np.linspace(0, 1, 100)
x1, x2 = np.meshgrid(x1, x2)
X_grid = np.vstack((x1.flatten(), x2.flatten())).T
y_grid = tf.reshape(custom_function(X_grid), x1.shape)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.contourf(x1, x2, y_grid, levels=20, cmap='viridis')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Función Custom')

# Graficando la salida del modelo
predictions = model.predict(X_grid)
predictions = tf.reshape(predictions, x1.shape)

plt.subplot(1, 2, 2)
plt.contourf(x1, x2, predictions, levels=20, cmap='viridis')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Salida del Modelo')

plt.tight_layout()
plt.show()


