import tensorflow as tf
import numpy as np
from PIL import Image
import io

IMG_SIZE = 224

def preprocess_image(image_bytes: bytes):
    """Preprocesa la imagen como en tu flujo de Colab: 224x224 + normalización 0-1."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = img_array / 255.0  # normalización
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
