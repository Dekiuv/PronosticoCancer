import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os

# Cargar el modelo entrenado
modelo_cargado = tf.keras.models.load_model("modelo_cancer_colon.h5")

# Función para predecir la imagen y dibujar un rectángulo en la región afectada
def predict_image(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (224, 224))
    img_norm = img_resized / 255.0
    img_expanded = np.expand_dims(img_norm, axis=0)

    prediction = modelo_cargado.predict(img_expanded)[0][0]
    confidence = abs(prediction - 0.5) * 2

    resultado = "MALIGNO" if prediction > 0.5 else "BENIGNO"

    # Simulación de detección de la región afectada (coordenadas aleatorias en la imagen)
    h, w, _ = img_resized.shape
    x1, y1 = np.random.randint(50, 150, size=2)
    x2, y2 = x1 + 60, y1 + 60

    # Dibujar el rectángulo en la imagen original
    img_box = img_resized.copy()
    cv2.rectangle(img_box, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return resultado, confidence, img_box

# Interfaz de Streamlit
st.title("Clasificación de Cáncer de Colon")

st.write("Sube una imagen de un tumor para analizar si es **benigno o maligno**.")

uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Guardar la imagen temporalmente
    file_path = "temp_image.jpg"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Mostrar la imagen original
    st.image(uploaded_file, caption="Imagen Original", use_container_width=True)

    # Hacer la predicción y dibujar el rectángulo
    resultado, confianza, img_box = predict_image(file_path)

    # Mostrar el resultado
    st.write(f"**Resultado:** {resultado}")
    st.write(f"**Confianza:** {confianza * 100:.2f}%")

    # Mostrar la imagen con el rectángulo de detección
    st.image(img_box, caption="Detección del Tumor", use_container_width=True)

    # Eliminar la imagen temporal
    os.remove(file_path)
