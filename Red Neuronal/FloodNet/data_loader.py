# data_loader.py

import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import to_categorical

def preprocess_image(image, mask, target_size=(128, 128)):
    # Redimensionar imagen
    image = tf.image.resize(image, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    # Redimensionar máscara
    mask = tf.image.resize(mask, target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image, mask

def load_data(img_dir, mask_dir, img_height, img_width, num_classes):
    # Carga de imágenes y máscaras
    images = []  # Lista para almacenar imágenes
    masks = []   # Lista para almacenar máscaras

    # Suponiendo que tienes una lista de nombres de archivos de imágenes y máscaras
    image_files = [...]  # Lista de nombres de archivos de imágenes
    mask_files = [...]   # Lista de nombres de archivos de máscaras

    for img_file, mask_file in zip(image_files, mask_files):
        # Cargar imagen
        img = tf.io.read_file(img_dir + img_file)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # Normalizar a [0,1]

        # Cargar máscara
        mask = tf.io.read_file(mask_dir + mask_file)
        mask = tf.image.decode_image(mask, channels=1)
        mask = tf.image.convert_image_dtype(mask, tf.uint8)  # Asegurar que la máscara sea entera

        # Preprocesar imagen y máscara
        img, mask = preprocess_image(img, mask, target_size=(img_height, img_width))

        images.append(img)
        masks.append(mask)

    # Convertir listas a arrays de NumPy
    images = np.array(images)
    masks = np.array(masks)

    # Convertir máscaras a one-hot encoding
    masks = to_categorical(masks, num_classes=num_classes)

    return images, masks