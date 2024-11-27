import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Set the current working directory to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Load the model
model_filename = 'testperfect.keras'
if not os.path.exists(model_filename):
    raise FileNotFoundError(f"The model file '{model_filename}' does not exist in the current directory: {script_dir}")

modelo = load_model(model_filename)

# Load and preprocess the CIFAR-10 dataset
(_, _), (x_test, y_test) = cifar10.load_data()

# Normalize the test data
mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2023, 0.1994, 0.2010])
x_test = (x_test.astype("float32") / 255.0 - mean) / std

y_test_categorical = to_categorical(y_test, 10)

# Make predictions
predicciones = modelo.predict(x_test[:10])

# CIFAR-10 class mapping
clases = ['avión', 'automóvil', 'pájaro', 'gato', 'ciervo', 'perro', 'rana', 'caballo', 'barco', 'camión']

# Variables to count correct and incorrect predictions
correctas = 0
incorrectas = 0

# Compare predictions with actual labels
for i in range(10):
    clase_predicha = np.argmax(predicciones[i])
    clase_real = y_test[i][0]
    
    if clase_predicha == clase_real:
        correctas += 1
    else:
        incorrectas += 1
    
    print(f'Imagen {i+1}:')
    print(f'    Clase predicha: {clases[clase_predicha]}')
    print(f'    Clase real: {clases[clase_real]}')
    print(f'    {"Correcto" if clase_predicha == clase_real else "Incorrecto"}\n')
    
    # Display the image
    plt.imshow(x_test[i] * std + mean)
    plt.title(f'Real: {clases[clase_real]}, Predicha: {clases[clase_predicha]}')
    plt.axis('off')
    plt.show()

# Calculate percentages
porcentaje_correctas = (correctas / 10) * 100
porcentaje_incorrectas = (incorrectas / 10) * 100

# Show final results
print(f'Porcentaje de predicciones correctas: {porcentaje_correctas}%')
print(f'Porcentaje de predicciones incorrectas: {porcentaje_incorrectas}%')