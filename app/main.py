from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.detection import PlateDetector
import base64
import cv2

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/plates/detect")
async def detect_plate_api(file: UploadFile = File(...)):
    image_bytes = await file.read()
    detector = PlateDetector()
    text, image = await detector.detect_from_bytes(image_bytes)

    _, buffer = cv2.imencode(".png", image)
    img_b64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse(content={"texto": text, "patente": img_b64})
