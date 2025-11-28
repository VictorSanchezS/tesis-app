import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
import base64


# ============================================================
# 1. ENCONTRAR LA ÚLTIMA CAPA CONVOLUCIONAL DEL MODELO
#    (versión simple y robusta)
# ============================================================

def get_last_conv_layer(model: keras.Model):
    """
    Devuelve la última capa Conv2D del modelo.
    Recorre todas las capas del modelo (de atrás hacia adelante)
    y devuelve la primera que sea instancia de keras.layers.Conv2D.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            # print(f"🔍 Última Conv2D: {layer.name}")
            return layer
    raise ValueError("No se encontró ninguna capa Conv2D en el modelo para Grad-CAM++.")


# ============================================================
# 2. GRAD-CAM++  (adaptado de tu BLOQUE 11)
# ============================================================

def grad_cam_pp(model: keras.Model, img_tensor: tf.Tensor) -> np.ndarray:
    """
    Calcula el mapa de calor Grad-CAM++ para una imagen.

    Parámetros
    ----------
    model : keras.Model
        Modelo Keras cargado desde .keras
    img_tensor : tf.Tensor
        Tensor de shape (1, 224, 224, 3), valores en [0,1]

    Retorna
    -------
    heatmap : np.ndarray
        Array 2D float32 en [0,1] con el mapa de calor.
    """
    # Asegurarnos de que es tensor float32
    img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)

    # Última capa conv
    last_conv_layer = get_last_conv_layer(model)

    # Modelo auxiliar: salida de la última conv + salida final
    grad_model = keras.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor, training=False)
        tape.watch(conv_out)

        # Forzar preds a tensor (soluciona el error en Windows/TensorFlow 2.20)
        preds = tf.convert_to_tensor(preds)

        loss = preds[:, 0]

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        # Si algo falla, devolvemos un mapa vacío
        h = img_tensor.shape[1]
        w = img_tensor.shape[2]
        return np.zeros((h, w), dtype=np.float32)

    # Convertir a numpy
    conv = conv_out[0].numpy()  # (H, W, C)
    g = grads[0].numpy()        # (H, W, C)

    g2 = g ** 2
    g3 = g ** 3

    # Evitar divisiones por cero
    denom = 2 * g2 + np.sum(g3 * conv, axis=(0, 1), keepdims=True)
    denom = np.where(denom != 0, denom, 1e-10)

    alpha = g2 / denom
    weights = np.sum(alpha * np.maximum(g, 0), axis=(0, 1))

    # heatmap: combinación lineal de los mapas de activación
    heatmap = np.tensordot(conv, weights, axes=([2], [0]))
    heatmap = np.maximum(heatmap, 0)

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap.astype(np.float32)


# ============================================================
# 3. SUPERPOSICIÓN DEL HEATMAP SOBRE LA IMAGEN ORIGINAL
# ============================================================

def superimpose_heatmap(img_uint8: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Combina la imagen original con el heatmap Grad-CAM++.

    img_uint8 : np.ndarray
        Imagen RGB uint8 (H, W, 3) en rango [0, 255]
    heatmap : np.ndarray
        Mapa 2D en [0,1]
    alpha : float
        Factor de mezcla del mapa de calor
    """
    h, w = img_uint8.shape[:2]

    # Redimensionar mapa a tamaño de la imagen
    heat_resized = cv2.resize(heatmap, (w, h))
    heat_uint8 = np.uint8(heat_resized * 255)

    # Colormap (similar al COLORMAP_TURBO)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    # Mezclar imagen original y mapa de calor
    overlay = cv2.addWeighted(img_uint8, 1 - alpha, heat_color, alpha, 0)
    return overlay


# ============================================================
# 4. FUNCIÓN PÚBLICA QUE USA FastAPI
# ============================================================

def generate_gradcampp(model: keras.Model, img_tensor: np.ndarray | tf.Tensor) -> str:
    """
    Función llamada desde app.main para generar la imagen XAI.

    - model: modelo Keras ya cargado
    - img_tensor: tensor/array (1, 224, 224, 3) normalizado [0,1]

    Devuelve:
        String base64 de un JPG con el overlay Grad-CAM++.
    """
    # Asegurarnos de que es tensor
    if not isinstance(img_tensor, tf.Tensor):
        img_tensor = tf.convert_to_tensor(img_tensor, dtype=tf.float32)

    # Extraer la imagen original en [0,255] uint8
    img_np = img_tensor[0].numpy()            # (224,224,3), float32 [0,1]
    img_uint8 = np.uint8(np.clip(img_np * 255.0, 0, 255))

    # Calcular mapa Grad-CAM++
    heatmap = grad_cam_pp(model, img_tensor)

    # Combinar
    overlay = superimpose_heatmap(img_uint8, heatmap)

    # Codificar a JPEG
    ok, buffer = cv2.imencode(".jpg", overlay)
    if not ok:
        raise RuntimeError("No se pudo codificar la imagen XAI a JPG.")

    # Convertir a base64
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return b64_str
