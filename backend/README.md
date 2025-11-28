# Backend – Detección de Anemia (FastAPI + MobileNetV2 + Grad-CAM++)

Este backend expone una API REST con FastAPI para:

- Cargar el modelo `modelo_mobilenetv2_anemia_final_XAI.keras`
- Recibir imágenes de uñas por `POST`
- Preprocesar la imagen (224x224, normalización 0-1)
- Devolver:
  - probabilidad de anemia
  - clase (1 = anémico, 0 = no anémico)
  - (opcional) imagen XAI (Grad-CAM++) en base64

## 1. Estructura de carpetas

Se asume que estás en:

`tesis-app/backend`

```bash
backend/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── load_model.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── preprocess.py
│   ├── xai/
│   │   ├── __init__.py
│   │   └── gradcampp.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── response.py
│   └── static/
├── modelo_mobilenetv2_anemia_final_XAI.keras   # (copiar aquí tu modelo)
├── requirements.txt
├── run.sh
└── README.md
```

## 2. Requisitos

- Python 3.10+
- `pip` instalado
- TensorFlow compatible con tu versión de Python

## 3. Instalación

```bash
cd backend
pip install -r requirements.txt
```

> Si usas un entorno virtual (recomendado):

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## 4. Añadir el modelo

Copia el archivo:

- `modelo_mobilenetv2_anemia_final_XAI.keras`

a la carpeta `backend/` (al mismo nivel que `requirements.txt`).

Si deseas usar otra ruta, puedes definir la variable de entorno:

```bash
export MODEL_PATH="ruta/a/tu/modelo.keras"
```

## 5. Ejecutar el servidor

### Opción 1: usando `run.sh` (Linux/Mac/WSL)

```bash
cd backend
chmod +x run.sh
./run.sh
```

### Opción 2: comando directo (Windows, Linux, Mac)

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

La API quedará disponible en:

- `http://localhost:8000`
- Documentación interactiva Swagger: `http://localhost:8000/docs`
- Documentación alternativa ReDoc: `http://localhost:8000/redoc`

## 6. Endpoints

### `GET /`

Verificar estado de la API.

**Respuesta ejemplo:**

```json
{
  "message": "API funcionando correctamente"
}
```

---

### `POST /predict`

Recibe una imagen y devuelve la predicción de anemia.

- Método: `POST`
- URL: `http://localhost:8000/predict`
- Parámetros:
  - `file`: archivo de imagen (form-data)
  - `xai`: boolean (query param, opcional). Si es `true`, genera Grad-CAM++.

#### Ejemplo con `curl`

```bash
curl -X POST \\
  -F "file=@unha.jpg" \\
  "http://localhost:8000/predict?xai=true"
```

#### Ejemplo de respuesta (sin XAI)

```json
{
  "probability": 0.87,
  "class": 1
}
```

#### Ejemplo de respuesta (con XAI)

```json
{
  "probability": 0.87,
  "class": 1,
  "xai_image_base64": "...cadena_base64..."
}
```

La cadena base64 representa una imagen `.jpg` con el mapa de calor sobre la uña.

## 7. Notas

- El preprocesamiento está alineado con tu flujo de Colab: redimensionado a 224x224 y normalización a [0, 1].
- El umbral de clasificación está fijado en 0.5:
  - `probability >= 0.5` → clase `1` (anémico)
  - `probability < 0.5` → clase `0` (no anémico)
- La implementación de Grad-CAM++ asume una arquitectura tipo CNN con una última capa convolucional antes de las capas densas. Si modificas la arquitectura, puede que necesites ajustar la lógica que detecta `last_conv_layer`.
