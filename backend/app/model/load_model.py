import tensorflow as tf
import os

# Ruta del modelo (se asume que lo colocarás en la carpeta backend/)
MODEL_PATH = os.getenv("MODEL_PATH", "modelo_mobilenetv2_anemia_final_XAI.keras")

print(f"Cargando modelo desde: {MODEL_PATH} ...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo cargado correctamente.")
