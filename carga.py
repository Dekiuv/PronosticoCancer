import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definir los parámetros
IMG_SIZE = (224, 224)  # Tamaño de la imagen
BATCH_SIZE = 32

# Generador de imágenes con aumento de datos
datagen = ImageDataGenerator(
    rescale=1./255,  # Normaliza los píxeles
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% entrenamiento, 20% validación
)

# Cargar las imágenes de entrenamiento
train_generator = datagen.flow_from_directory(
    'datos',  # Directorio base que contiene 'maligno/' y 'benigno/'
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',  # Clasificación binaria (0 = benigno, 1 = maligno)
    subset='training'
)

# Cargar las imágenes de validación
val_generator = datagen.flow_from_directory(
    'datos',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Cargar un modelo preentrenado (MobileNetV2)
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Congelar capas del modelo base
base_model.trainable = False

# Definir la arquitectura de la red neuronal
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),  # Para evitar sobreajuste
    tf.keras.layers.Dense(1, activation='sigmoid')  # Salida binaria (0 = benigno, 1 = maligno)
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Mostrar el resumen del modelo
model.summary()