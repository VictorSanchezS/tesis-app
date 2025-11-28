# 📘 Guía de instalación -- Proyecto tesis-app

Este documento explica cómo clonar el proyecto y ejecutar el backend
(FastAPI) en cualquier dispositivo.

## ✅ 1. Clonar el repositorio

Abre la terminal (CMD, PowerShell o Git Bash) y ejecuta:

`git clone https://github.com/VictorSanchezS/tesis-app.git`

Esto creará la siguiente estructura:

tesis-app/ ├── backend/ ├── frontend/ ├── .gitignore └── README.md

## ✅ 2. Entrar al proyecto 

`cd tesis-app`

## ✅ 3. Preparar el backend 

3.1 Entrar a la carpeta del backend `cd backend`

3.2 Crear el entorno virtual (Windows) `python -m venv venv`

3.3 Activar el entorno virtual 

*** Copiar y pega en consola:**

*** venv`\Scripts`{=tex}`\activate`{=tex} ***

O

`.\venv\Scripts\Activate.ps1`

Si la terminal muestra (venv) significa que está activado.

3.4 Instalar dependencias `pip install -r requirements.txt`

3.5 Ejecutar FastAPI `uvicorn app.main:app --reload`

El backend estará disponible en:

👉 http://127.0.0.1:8000

Y la documentación interactiva:

👉 http://127.0.0.1:8000/docs
