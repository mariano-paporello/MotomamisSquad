from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.schemas import PlateResponse
from app.detection import PlateDetector
import base64
import cv2

router = APIRouter()

@router.post("/detect", response_model=PlateResponse)
async def detect_plate_endpoint(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="The file must be an image.")

    image_bytes = await file.read()
    detector = PlateDetector()

    try:
        plate_text, cropped_image = await detector.detect_from_bytes(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not plate_text or cropped_image is None:
        raise HTTPException(status_code=404, detail="No plate detected.")

    _, buffer = cv2.imencode(".png", cropped_image)
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    data_uri = f"data:image/png;base64,{img_base64}"

    return PlateResponse(plate=plate_text, image=data_uri)
