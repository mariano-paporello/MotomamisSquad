from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.schemas import PlateResponse
from app.detection import detect_plate_wrapper
import numpy as np
import base64
import cv2

router = APIRouter()

@router.post("/detect", response_model=PlateResponse)
async def detect_plate_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")

    image_bytes = await file.read()

    try:
        plate_text, crop_img = await detect_plate_wrapper(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not plate_text or crop_img is None:
        raise HTTPException(status_code=404, detail="No se detectó ninguna patente.")

    # Codificar imagen recortada a base64
    _, buffer = cv2.imencode(".png", crop_img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    data_uri = f"data:image/png;base64,{img_base64}"

    return PlateResponse(plate=plate_text, image=data_uri)
