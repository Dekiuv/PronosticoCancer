import carga as ca

# Entrenar el modelo
EPOCHS = 10

history = ca.model.fit(
    ca.train_generator,
    epochs=EPOCHS,
    validation_data=ca.val_generator
)

# Evaluar el modelo
loss, accuracy = ca.model.evaluate(ca.val_generator)
print(f'Precisión en validación: {accuracy:.2f}')

# Guardar el modelo entrenado
ca.model.save("modelo_cancer_colon.h5")