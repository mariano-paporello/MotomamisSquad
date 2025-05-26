import sys
import os
import cv2
import torch
import numpy as np
import re
from pathlib import Path
from paddleocr import PaddleOCR

# --- Configurar rutas ---
BASE_DIR = Path(__file__).resolve().parent.parent
YOLOV5_DIR = BASE_DIR / 'yolov5'
sys.path.insert(0, str(YOLOV5_DIR))  # Agrega yolov5 al sys.path

from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# --- Inicializar PaddleOCR con rutas explícitas ---
ocr = PaddleOCR(use_angle_cls=True, lang='en')
device = select_device('cpu')
modelo_path = BASE_DIR / 'model' / 'best_windows_safe.pt'
model = attempt_load(str(modelo_path), device)
model.eval()
todos_los_resultados = []

# --- Funciones auxiliares ---
def formatear_patente(texto):
    texto = texto.upper().replace(" ", "").replace("\n", "")
    return re.sub(r'[^A-Z0-9]', '', texto)

def es_patente_valida(txt):
    return bool(
        re.match(r'^[A-Z]{3}\d{3}$', txt) or
        re.match(r'^[A-Z]{2}\d{3}[A-Z]{2}$', txt)
    )

def aplicar_nitidez_y_clahe(img):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (9, 9), 10.0)
    sharp = cv2.addWeighted(gris, 1.5, blur, -0.5, 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(sharp)

def escalar_imagen(img, factor=3.0):
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

def ocr_paddle_con_confianza(path):
    img = cv2.imread(path)
    if img is None:
        print(f"❌ No se pudo leer la imagen para OCR: {path}")
        return "", 0.0
    resultado = ocr.ocr(img)
    posibles = []
    if resultado:
        for linea in resultado:
            if linea is not None:
                for bbox in linea:
                    if bbox and len(bbox) == 2 and isinstance(bbox[1], tuple):
                        txt, conf = bbox[1]
                        texto_limpio = formatear_patente(txt)
                        if conf > 0.5:
                            print(f"🔍 Texto: {texto_limpio} (confianza: {conf:.2f})")
                        if es_patente_valida(texto_limpio):
                            return texto_limpio, conf
                        posibles.append((texto_limpio, conf))
    if posibles:
        return posibles[0]
    return "", 0.0

# --- FUNCIÓN PRINCIPAL PARA API ---
async def detect_plate_wrapper(image_bytes: bytes):
    carpeta_resultados = BASE_DIR / 'resultados'
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    todos_los_resultados.clear()

    arr = np.frombuffer(image_bytes, np.uint8)
    imagen = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if imagen is None:
        raise ValueError("❌ No se pudo decodificar la imagen enviada.")

    h, w = imagen.shape[:2]
    img_resized = cv2.resize(imagen, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    if pred is None or len(pred) == 0:
        raise ValueError("No se detectó ninguna patente.")

    xyxy = pred[0][:4].cpu().numpy()
    x1, y1, x2, y2 = (xyxy * np.array([w/640, h/640, w/640, h/640])).astype(int)
    patente_recortada = imagen[y1:y2, x1:x2]
    path_crop = str(carpeta_resultados / '1_patente_recortada.png')
    cv2.imwrite(path_crop, patente_recortada)

    # --- Intentos de OCR ---
    def intentar_ocr(img, nombre_archivo):
        path = str(carpeta_resultados / nombre_archivo)
        cv2.imwrite(path, img)
        texto, conf = ocr_paddle_con_confianza(path)
        print(f"🔁 OCR {nombre_archivo}: {texto} (conf: {conf:.2f})")
        todos_los_resultados.append((texto, conf))

    original_esc = escalar_imagen(patente_recortada)
    intentar_ocr(original_esc, '1a_patente_escalada.png')

    nitidez_clahe = aplicar_nitidez_y_clahe(original_esc)
    intentar_ocr(nitidez_clahe, '2_patente_nitidez_clahe.png')

    invertida = cv2.bitwise_not(cv2.cvtColor(patente_recortada, cv2.COLOR_BGR2GRAY))
    intentar_ocr(escalar_imagen(invertida), '3_invertida_grises.png')

    h, w = original_esc.shape[:2]
    nuevo_ancho = int(w * 1.5)
    nuevo_alto = h
    pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    pts2 = np.float32([[0, 0], [nuevo_ancho - 1, 0], [0, nuevo_alto - 1], [nuevo_ancho - 1, nuevo_alto - 1]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warp = cv2.warpPerspective(original_esc, M, (nuevo_ancho, nuevo_alto))
    intentar_ocr(escalar_imagen(warp), '3a_warp_escalada.png')

    nitidez_clahe_warp = aplicar_nitidez_y_clahe(warp)
    intentar_ocr(nitidez_clahe_warp, '4_patente_nitidez_clahe_warp.png')

    warp_inv_gris = cv2.bitwise_not(cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY))
    intentar_ocr(escalar_imagen(warp_inv_gris), '5_invertida_escalada.png')

    candidatos_validos = [
        (t, c) for t, c in todos_los_resultados
        if re.fullmatch(r'[A-Z]{3}\d{3}', t) or re.fullmatch(r'[A-Z]{2}\d{3}[A-Z]{2}', t)
    ]
    if candidatos_validos:
        mejor_valido, conf_valido = max(candidatos_validos, key=lambda x: x[1])
        print(f"✅ Texto detectado válido: {mejor_valido} (conf: {conf_valido:.2f})")
        return mejor_valido, patente_recortada
    elif todos_los_resultados:
        mejor_texto, mejor_conf = max(todos_los_resultados, key=lambda x: x[1])
        print(f"⚠️ No se detectó una patente con formato válido, pero el mejor intento fue: {mejor_texto} (conf: {mejor_conf:.2f})")
        return mejor_texto, patente_recortada
    else:
        print("⚠️ No se detectó ningún texto.")
        return "", patente_recortada

