from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app import detection  
from app.detection import detect_plate_wrapper
import cv2
import numpy as np
import base64
from app.schemas import PlateResponse

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/plates/detect", response_model=PlateResponse)
async def detect_plate_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    lista_patentes, crop_img = await detect_plate_wrapper(image_bytes)

    # Asegurarse de que sea lista, incluso si solo hay una
    if isinstance(lista_patentes, str):
        lista_patentes = [lista_patentes]

    # Convertir imagen recortada a base64
    _, buffer = cv2.imencode('.png', crop_img)
    data_uri = base64.b64encode(buffer).decode('utf-8')

    return PlateResponse(plate=lista_patentes, image=data_uri)
