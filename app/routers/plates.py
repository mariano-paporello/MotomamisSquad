from fastapi import APIRouter, UploadFile, File, HTTPException
from app.logic.processor import process_image
from fastapi.responses import JSONResponse
from app import schemas
import numpy as np
import cv2
import base64

router = APIRouter()

@router.post("/detect", response_model=schemas.PlateResponse)
async def detect_plate(file: UploadFile = File(...)):
    # Validar tipo
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="No se pudo leer la imagen enviada.")

    try:
        plate_text, plate_img = process_image(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Codificar recorte a PNG + Base64
    success, buf = cv2.imencode('.png', plate_img)
    if not success:
        raise HTTPException(status_code=500, detail="Error al codificar la imagen recortada.")
    b64 = base64.b64encode(buf).decode('utf-8')
    data_uri = f"data:image/png;base64,{b64}"

    return schemas.PlateResponse(plate=plate_text, image=data_uri)