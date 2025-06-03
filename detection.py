import sys
import os
import cv2
import torch
import numpy as np
import re
from pathlib import Path
from paddleocr import PaddleOCR
from scipy.signal import wiener
from skimage import exposure, img_as_ubyte
from skimage.color import rgb2gray

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

def resaltar_letras_negras(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    contraste = clahe.apply(gray)
    _, binarizada = cv2.threshold(contraste, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    reforzada = cv2.dilate(binarizada, kernel, iterations=1)
    final = cv2.bitwise_not(reforzada)
    return final

def aplicar_wiener(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filtrada = wiener(gray)
    filtrada = np.uint8(np.clip(filtrada, 0, 255))
    return filtrada

def mejorar_contraste(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p2, p98 = np.percentile(gray, (2, 98))
    estirada = exposure.rescale_intensity(gray, in_range=(p2, p98))
    adapthist = exposure.equalize_adapthist(estirada, clip_limit=0.03)
    final = img_as_ubyte(adapthist)
    return final

def preprocesar(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)
    _, binarizada = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binarizada) < 127:
        binarizada = cv2.bitwise_not(binarizada)
    return binarizada

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

def corregir_patente_por_formato(texto):
    texto = texto.upper().replace(" ", "").replace("\n", "")
    if re.fullmatch(r'[A-Z0-9]{2}[A-Z0-9]{3}[A-Z0-9]{2}', texto):
        letras1 = texto[:2].replace("1", "I").replace("0", "O").replace("4", "A").replace("8", "B").replace("5", "S").replace("6", "G")
        numeros = texto[2:5].replace("I", "1").replace("O", "0").replace("S", "5").replace("A", "4").replace("B", "8")
        letras2 = texto[5:].replace("1", "I").replace("0", "O").replace("S", "5").replace("A", "4").replace("B", "8").replace("6", "G")
        return letras1 + numeros + letras2
    if re.fullmatch(r'[A-Z0-9]{3}[A-Z0-9]{3}', texto):
        letras = texto[:3].replace("1", "I").replace("0", "O").replace("4", "A").replace("8", "B").replace("5", "S").replace("6", "G")
        numeros = texto[3:].replace("I", "1").replace("O", "0").replace("S", "5").replace("A", "4").replace("B", "8")
        return letras + numeros
    return texto

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
    
    wiener_filtrada = aplicar_wiener(original_esc)
    intentar_ocr(wiener_filtrada, '2_patente_wiener.png')

    nitidez_clahe = aplicar_nitidez_y_clahe(original_esc)
    intentar_ocr(nitidez_clahe, '3_patente_nitidez_clahe.png')
   
    original_reforzada = resaltar_letras_negras(original_esc)
    intentar_ocr(original_reforzada, '4_patente_escalada_reforzada.png')

    mejorada = mejorar_contraste(original_esc)
    intentar_ocr(mejorada, '5_patente_mejorada_contraste.png')

    invertida = cv2.bitwise_not(patente_recortada)
    intentar_ocr(escalar_imagen(invertida), '6_invertida_escalada.png')
    
    # OCR 7: preprocesada escalada
    preproc = preprocesar(patente_recortada)
    intentar_ocr(escalar_imagen(preproc), '7_preproc_escalada.png')

    h, w = original_esc.shape[:2]
    nuevo_ancho = int(w * 2)
    nuevo_alto = h
    pts1 = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    pts2 = np.float32([[0, 0], [nuevo_ancho - 6, 0], [0, nuevo_alto - 6], [nuevo_ancho - 4, nuevo_alto - 5]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warp = cv2.warpPerspective(original_esc, M, (nuevo_ancho, nuevo_alto))
    intentar_ocr(escalar_imagen(warp), '8_warp_escalada.png')

    nitidez_clahe_warp = aplicar_nitidez_y_clahe(warp)
    intentar_ocr(nitidez_clahe_warp, '9_patente_nitidez_clahe_warp.png')

    invertida_warp = cv2.bitwise_not(warp)
    intentar_ocr(escalar_imagen(invertida_warp), '10_invertida_escalada_warp.png')

    original_reforzada_warp = resaltar_letras_negras(warp)
    intentar_ocr(original_reforzada_warp, '11_patente_escalada_reforzada.png')

    wiener_filtrada_warp = aplicar_wiener(warp)
    intentar_ocr(wiener_filtrada_warp, '12_patente_wiener.png')

    mejorada_warp = mejorar_contraste(warp)
    intentar_ocr(mejorada_warp, '13_patente_mejorada_contraste.png')  
    
    # OCR 13: preprocesada escalada
    preproc_warp = preprocesar(warp)
    intentar_ocr(escalar_imagen(preproc_warp), '14_preproc_escalada.png')

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
        corregido = corregir_patente_por_formato(mejor_texto)
        print(f"⚠️ No se detectó una patente con formato válido, pero el mejor intento fue: {corregido} (conf: {mejor_conf:.2f})")
        return corregido, patente_recortada
    else:
        print("⚠️ No se detectó ningún texto.")
        return "", patente_recortada
