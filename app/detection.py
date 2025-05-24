import sys
import os
import cv2
import torch
import numpy as np
import re
from pathlib import Path
from paddleocr import PaddleOCR

# Configurar rutas e imports yolov
sys.path.insert(0, r'C:\Users\Ginette Henriquez\Downloads\TP\yolov5')
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# Inicializar OCR y modelo
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
device = select_device('cpu')
modelo_path = Path(__file__).resolve().parent.parent / 'model' / 'best_windows_safe.pt'
model = attempt_load(str(modelo_path), device)
model.eval()
todos_los_resultados = []

# --- Funciones auxiliares ---
def formatear_patente(texto):
    texto = texto.upper().replace(" ", "").replace("\n", "")
    return re.sub(r'[^A-Z0-9]', '', texto)

def es_patente_valida(txt):
    return bool(
        re.match(r'^[A-Z]{3}\d{3}$', txt) or         # ABC123 1995 A 2016
        re.match(r'^[A-Z]{2}\d{3}[A-Z]{2}$', txt) # AB123CD ACTUALMENTE
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
    resultado = ocr.ocr(img, cls=True)
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

# --- FUNCIÓN PARA LA API ---
async def detect_plate_wrapper(image_bytes: bytes):
    carpeta_resultados = Path(__file__).resolve().parent.parent / 'resultados'
    carpeta_resultados.mkdir(parents=True, exist_ok=True)

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
    path_crop = os.path.join(carpeta_resultados, '1_patente_recortada.png')
    cv2.imwrite(path_crop, patente_recortada)

    def intentar_ocr(img, nombre_archivo):
        path = os.path.join(carpeta_resultados, nombre_archivo)
        cv2.imwrite(path, img)
        texto, conf = ocr_paddle_con_confianza(path)
        print(f"🔁 OCR {nombre_archivo}: {texto} (conf: {conf:.2f})")
        todos_los_resultados.append((texto, conf))  # ← guarda aunque no sea válido

    # OCR 2: original escalado
    original_esc = escalar_imagen(patente_recortada)
    resultado = intentar_ocr(original_esc, '2_patente_escalada.png')
    if resultado:
        return resultado, patente_recortada

    # OCR 3: original escalado + nitidez + clahe
    nitidez_clahe = aplicar_nitidez_y_clahe(original_esc)
    resultado = intentar_ocr(nitidez_clahe, '3_patente_nitidez_clahe.png')

    # OCR 4: invertido grises
    invertidagris = cv2.bitwise_not(cv2.cvtColor(patente_recortada, cv2.COLOR_BGR2GRAY))
    resultado = intentar_ocr(escalar_imagen(invertidagris), '4_invertida_escalada_gris.png')

    # OCR 5: warp escalado mejorado con ancho extendido
    h, w = patente_recortada.shape[:2]
    pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    nuevo_ancho = int(w * 1.5)
    nuevo_alto = h
    pts2 = np.float32([[0, 0], [nuevo_ancho - 1, 0], [0, nuevo_alto - 1], [nuevo_ancho - 1, nuevo_alto - 1]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warp = cv2.warpPerspective(patente_recortada, M, (nuevo_ancho, nuevo_alto))
    resultado = intentar_ocr(escalar_imagen(warp), '5_warp_escalada.png')
    
    # OCR 6: wrap + nitidez + clahe
    nitidez_clahe_warp = aplicar_nitidez_y_clahe(warp)
    resultado = intentar_ocr(nitidez_clahe_warp, '6_patente_warp_nitidez_clahe.png')
    
    # OCR 7: warp invertida escalada grises
    warp_inv = cv2.bitwise_not(cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY))
    resultado = intentar_ocr(escalar_imagen(warp_inv), '7_warp_invertida_escalada_gris.png')

    if todos_los_resultados:
        candidatos_validos = [ 
                              (t, c) for t, c in todos_los_resultados
                              if re.fullmatch(r'[A-Z]{3}\d{3}', t) or re.fullmatch(r'[A-Z]{2}\d{3}[A-Z]{2}', t)
        ]

        if candidatos_validos:
            mejor_valido, conf_valido = max(candidatos_validos, key=lambda x: x[1])
            print(f"✅ Texto detectado válido: {mejor_valido} (conf: {conf_valido:.2f})")
            return mejor_valido, patente_recortada
        else:
            mejor_texto, mejor_conf = max(todos_los_resultados, key=lambda x: x[1])
            print(f"⚠️ No se detectó una patente con formato válido, pero el mejor intento fue: {mejor_texto} (conf: {mejor_conf:.2f})")
            return mejor_texto, patente_recortada
    else:
        raise ValueError("⚠️ No se detectó ningún texto.")
