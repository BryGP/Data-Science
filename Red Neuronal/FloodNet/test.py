import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import load_model
from data_loader import load_data
from constants import img_height, img_width, num_classes
from visualize import decode_mask

# Cargar datos de prueba
x_test, y_test = load_data('D:/FloodNet-Supervised_v1.0/test/test-org-img', 'D:/FloodNet-Supervised_v1.0/test/test-label-img', img_height, img_width, num_classes)

# Cargar modelo entrenado
model = load_model('floodnet_model_best.keras')

#Normalizar las imágenes
x_test = np.array(x_test, dtype=np.float32) / 255.0

# Generar predicciones
predictions = model.predict(x_test)
predictions_classes = np.argmax(predictions, axis=-1)
print("Forma de las predicciones: ", predictions.shape)
print("Ejemplo de prediccion: ", np.unique(predictions_classes))

# Visualizar resultados
for i in range(5):  # Mostrar 5 imágenes
    plt.figure(figsize=(10, 5))
    
    # Imagen original
    plt.subplot(1, 3, 1)
    plt.imshow(x_test[i])
    plt.title('Imagen Original')
    
    # Máscara verdadera
    plt.subplot(1, 3, 2)
    plt.imshow(y_test[i].argmax(axis=-1), cmap='jet')
    plt.title('Máscara Verdadera')

    # Predicción
    plt.subplot(1, 3, 3)
    plt.imshow(predictions_classes[i], cmap='jet')
    plt.title('Predicción')
    plt.show()

decode_prediction = decode_mask(predictions_classes[0])
plt.imshow(decode_prediction)
plt.title('Predicción Decodificada')
plt.show()