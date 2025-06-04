from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app import detection  # o importá tus routers reales
import cv2
import numpy as np
import base64

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/plates/detect")
async def detect_plate_api(file: UploadFile = File(...)):
    image_bytes = await file.read()

    from app.detection import detect_plate_wrapper  # asegurate que esté ahí
    texto, imagen = await detect_plate_wrapper(image_bytes)

    # Convertir imagen recortada a base64 para el frontend
    _, buffer = cv2.imencode(".png", imagen)
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse(content={"texto": texto, "patente": img_b64})
