import cv2
import numpy as np

IMG_SIZE = 224

def aplicar_clahe(imagen):
    """
    Aplica CLAHE en el canal Y del espacio YCrCb.
    Mismo preprocesamiento usado durante el entrenamiento en Colab.
    """
    ycrcb = cv2.cvtColor(imagen, cv2.COLOR_RGB2YCrCb)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Aplicar solo a canal Y
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

def preprocess_image(image_bytes: bytes):
    """
    Pipeline EXACTO del entrenamiento:
    - cv2.imdecode (bytes → BGR)
    - BGR → RGB
    - resize 224x224
    - aplicar CLAHE
    - normalizar [0,1]
    """
    # Leer imagen desde bytes usando OpenCV
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Error al decodificar la imagen")

    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Redimensionar como en entrenamiento
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Aplicar CLAHE
    img = aplicar_clahe(img)

    # Normalizar a [0, 1]
    img = img.astype("float32") / 255.0

    # Expandir batch
    img = np.expand_dims(img, axis=0)

    return img
