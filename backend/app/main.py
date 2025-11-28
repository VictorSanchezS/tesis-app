from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.model.load_model import model
from app.utils.preprocess import preprocess_image
from app.xai.gradcampp import generate_gradcampp

import traceback

app = FastAPI(
    title="Anemia Detection API",
    description="API para detección de anemia con MobileNetV2 y Grad-CAM++",
    version="1.0.0",
)

# CORS (para frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"message": "API funcionando correctamente"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), xai: bool = False):

    try:
        # Leer bytes de la imagen
        image_bytes = await file.read()

        # Preprocesar imagen (224x224, normalización)
        img_tensor = preprocess_image(image_bytes)

        # Predicción
        prob = float(model.predict(img_tensor)[0][0])
        clazz = 1 if prob >= 0.5 else 0

        # Preparar respuesta base
        response = {
            "probability": prob,
            "class": clazz,
        }

        # XAI opcional
        if xai:
            print("➡️ Ejecutando XAI...")
            xai_image = generate_gradcampp(model, img_tensor)
            print("✔️ XAI generado correctamente")
            response["xai_image_base64"] = xai_image
        else:
            response["xai_image_base64"] = None

        # ESTE RETURN ES EL QUE FALTABA
        return JSONResponse(content=response)

    except Exception as e:
        print("\n\n========== ERROR INTERNO EN PREDICT/XAI ==========")
        traceback.print_exc()
        print("==========================================\n\n")
        return JSONResponse(content={"error": str(e)}, status_code=500)
