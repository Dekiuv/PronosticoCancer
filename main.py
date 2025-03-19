import tensorflow as tf
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog

# Función para hacer predicciones con el modelo cargado
def predict_image(image_path, model):
    img = cv2.imread(image_path)  # Cargar imagen
    img = cv2.resize(img, (224, 224))  # Redimensionar
    img = img / 255.0  # Normalizar
    img = np.expand_dims(img, axis=0)  # Añadir dimensión batch
    
    prediction = model.predict(img)[0][0]  # Obtener el valor de predicción
    if prediction > 0.5:
        print("El tumor es MALIGNO")
    else:
        print("El tumor es BENIGNO")

# Cargar el modelo guardado
modelo_cargado = tf.keras.models.load_model("modelo_cancer_colon.h5")

# Función para que el usuario suba una imagen y se haga la predicción
def subir_y_predecir():
    root = tk.Tk()
    root.withdraw()  # Ocultar la ventana principal
    
    image_path = filedialog.askopenfilename(title="Selecciona una imagen", filetypes=[("Imagenes", "*.jpg;*.jpeg;*.png")])
    if image_path:
        predict_image(image_path, modelo_cargado)
    else:
        print("No se seleccionó ninguna imagen.")

# Llamar a la función para que el usuario suba una imagen
subir_y_predecir()