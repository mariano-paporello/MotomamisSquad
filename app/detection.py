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
from ultralytics import YOLO

# --- Configurar rutas ---
BASE_DIR = Path(__file__).resolve().parent.parent
YOLOV5_DIR = BASE_DIR / 'yolov5'
sys.path.insert(0, str(YOLOV5_DIR))  # Agrega yolov5 al sys.path

from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device

# Inicializar OCR y modelo
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
modelo_path = Path(__file__).resolve().parent.parent / 'model' / 'license_plate_detector.pt'
model = YOLO(modelo_path)
todos_los_resultados = []

# --- Funciones auxiliares ---
class PlateTextProcessor:
    @staticmethod
    def formatear_patente(texto):
        texto = texto.upper().replace(" ", "").replace("\n", "")
        return re.sub(r'[^A-Z0-9]', '', texto)

    @staticmethod
    def es_patente_valida(txt):
        return bool(
            re.match(r'^[A-Z]{3}\d{3}$', txt) or
            re.match(r'^[A-Z]{2}\d{3}[A-Z]{2}$', txt)
        )

    @staticmethod
    def corregir_prefijo_patente(txt):
        if re.fullmatch(r'[A-Z]{2}\d{3}[A-Z]{2}', txt):
            if not txt.startswith('A'):
                corregida = 'A' + txt[1:]
                print(f"🔧 Corregido prefijo 7 caracteres: {txt} → {corregida}")
                return corregida
        return txt

    @staticmethod
    def corregir_patente_por_formato(texto):
        texto = PlateTextProcessor.formatear_patente(texto)
        if len(texto) == 7:
            letras1 = texto[:2].replace("4", "A").replace("1", "I").replace("0", "O").replace("8", "B").replace("6", "G").replace("5", "S")
            numeros = texto[2:5].replace("I", "1").replace("O", "0").replace("S", "5").replace("B", "8").replace("G", "6")
            letras2 = texto[5:].replace("1", "I").replace("0", "O").replace("8", "B").replace("6", "G").replace("5", "S")
            return letras1 + numeros + letras2
        if len(texto) == 6:
            letras = texto[:3].replace("4", "A").replace("1", "I").replace("0", "O").replace("8", "B").replace("6", "G").replace("5", "S")
            numeros = texto[3:].replace("I", "1").replace("O", "0").replace("S", "5").replace("B", "8").replace("G", "6")
            return letras + numeros
        return texto
    
class ImagePreprocessor:
    @staticmethod
    def escalar_imagen(img, factor=3.0):
        return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def escalar_con_denoise_y_contraste(img):
        escalada = ImagePreprocessor.escalar_imagen(img, factor=2.5)
        denoised = cv2.fastNlMeansDenoisingColored(escalada, None, 10, 10, 7, 21)
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(10, 8))
        cl = clahe.apply(l)
        final = cv2.merge((cl, a, b))
        return cv2.cvtColor(final, cv2.COLOR_LAB2BGR)

    @staticmethod
    def denoise(img):
        return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)

    @staticmethod
    def mejorar_contraste(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        p2, p98 = np.percentile(gray, (2, 98))
        estirada = exposure.rescale_intensity(gray, in_range=(p2, p98))
        adapthist = exposure.equalize_adapthist(estirada, clip_limit=0.03)
        return img_as_ubyte(adapthist)

    @staticmethod
    def preprocesar(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(gray)
        _, binarizada = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(binarizada) < 127:
            binarizada = cv2.bitwise_not(binarizada)
        return binarizada

def ocr_paddle_con_confianza(path):
    img = cv2.imread(path)
    resultado = ocr.ocr(img, cls=True)
    posibles = []

    if resultado:
        for linea in resultado:
            if linea:
                for bbox in linea:
                    if bbox and len(bbox) == 2 and isinstance(bbox[1], tuple):
                        txt, conf = bbox[1]
                        texto_limpio = PlateTextProcessor.formatear_patente(txt)
                        print(f"🔍 Texto: {texto_limpio} (confianza: {conf:.2f})")
                        posibles.append((texto_limpio, conf))

    return posibles if posibles else [("", 0.0)]


def recorte_por_proyecciones_img(img: np.ndarray) -> np.ndarray:
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    mejorada = clahe.apply(gris)

    kernel_prewitt_v = np.array([[1, 0, -1],
                                  [1, 0, -1],
                                  [1, 0, -1]])
    bordes = cv2.filter2D(mejorada, -1, kernel_prewitt_v)
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    apertura = cv2.morphologyEx(bordes, cv2.MORPH_OPEN, kernel_vertical)

    proy_h = np.sum(apertura, axis=1)
    umbral_h = 0.2 * np.max(proy_h)
    filas_validas = np.where(proy_h > umbral_h)[0]

    proy_v = np.sum(apertura, axis=0)
    umbral_v = 0.2 * np.max(proy_v)
    columnas_validas = np.where(proy_v > umbral_v)[0]

    if len(filas_validas) > 0 and len(columnas_validas) > 0:
        ymin, ymax = filas_validas[0], filas_validas[-1]
        xmin, xmax = columnas_validas[0], columnas_validas[-1]
        return img[ymin:ymax, xmin:xmax]
    else:
        print("⚠️ No se pudo detectar una zona clara con caracteres.")
        return img
    
# --- FUNCIÓN PARA LA API ---
async def detect_plate_wrapper(image_bytes: bytes):
    carpeta_resultados = Path(__file__).resolve().parent.parent / 'resultados'
    carpeta_resultados.mkdir(parents=True, exist_ok=True)
    todos_los_resultados = []

    # --- Función para ejecutar OCR y guardar resultados ---
    def intentar_ocr(img, nombre_archivo):
        path = os.path.join(carpeta_resultados, nombre_archivo)
        cv2.imwrite(path, img)
        resultados = ocr_paddle_con_confianza(path)
        for texto, conf in resultados:
            todos_los_resultados.append((texto, conf))

    # --- Decodificar imagen ---
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        print("❌ Imagen inválida.")
        return [], img

    # --- 
    results = model(img)[0]

    if results.boxes:
        x1, y1, x2, y2 = map(int, results.boxes[0].xyxy[0])
        patente = img[y1:y2, x1:x2]
        cv2.imwrite(str(carpeta_resultados / '1_patente_recortada.png'), patente)

        # --- Flujo de OCR sobre el recorte ---
        original_esc = ImagePreprocessor.escalar_imagen(patente)
        intentar_ocr(original_esc, '1a_patente_escalada.png')

        recortada_xy = recorte_por_proyecciones_img(original_esc)
        intentar_ocr(recortada_xy, '2_patente_recorte_xy.png')

        mejorada = ImagePreprocessor.mejorar_contraste(recortada_xy)
        intentar_ocr(mejorada, '3_patente_mejorada_contraste.png')

        pruebita = ImagePreprocessor.escalar_con_denoise_y_contraste(recortada_xy)
        intentar_ocr(ImagePreprocessor.escalar_imagen(pruebita), '4_escalar_con_denoise_y_contraste.png')

        invertidita = cv2.bitwise_not(recortada_xy)
        intentar_ocr(ImagePreprocessor.escalar_imagen(invertidita), '5_invertida_grises.png')

        preproci = ImagePreprocessor.preprocesar(recortada_xy)
        intentar_ocr(ImagePreprocessor.escalar_imagen(preproci), '6_preproc_escalada.png')

        proces = ImagePreprocessor.denoise(recortada_xy)
        intentar_ocr(proces, '7_patente_denoising_mejorada.png')

        # Warp + variantes
        h, w = recortada_xy.shape[:2]
        nuevo_ancho = int(w * 2)
        pts1 = np.float32([[0, 0], [w - 2, 4], [8, h - 6], [w - 8, h - 6]])
        pts2 = np.float32([[0, 0], [nuevo_ancho - 6, 4], [8, h - 6], [nuevo_ancho - 8, h - 4]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warp = cv2.warpPerspective(recortada_xy, M, (nuevo_ancho, h))
        intentar_ocr(ImagePreprocessor.escalar_imagen(warp), '8_warp_escalada.png')

        denoisingwarp = ImagePreprocessor.escalar_con_denoise_y_contraste(warp)
        intentar_ocr(denoisingwarp, '9_patente_denoising_mejorada.png')

        warpi = ImagePreprocessor.escalar_con_denoise_y_contraste(warp)
        intentar_ocr(ImagePreprocessor.escalar_imagen(warpi), '10_escalar_con_denoise_y_contraste.png')

        warp_inv_gris = cv2.bitwise_not(warp)
        intentar_ocr(ImagePreprocessor.escalar_imagen(warp_inv_gris), '11_invertida_escalada.png')

        mejoradarr = ImagePreprocessor.mejorar_contraste(warp)
        intentar_ocr(mejoradarr, '12_patente_mejorada_contraste.png')

        preproc = ImagePreprocessor.preprocesar(warp)
        intentar_ocr(ImagePreprocessor.escalar_imagen(preproc), '13_preproc_escalada.png')

    else:
        # --- Fallback OCR sobre imagen completa ---
        print("❌ No se detectó patente. Fallback OCR imagen completa...")
        preprocessed = ImagePreprocessor.preprocesar(img)
        intentar_ocr(ImagePreprocessor.escalar_imagen(preprocessed), 'fallback_full_image.png')
        patente = img

    # Procesamiento después de recortes y OCR de los recortes
    validos = [
        (t, c) for t, c in todos_los_resultados
        if re.fullmatch(r'[A-Z]{3}\d{3}', t) or re.fullmatch(r'[A-Z]{2}\d{3}[A-Z]{2}', t)
    ]

    if not validos:
        # No hubo texto válido en recortes: fallback OCR sobre imagen completa
        print("⚠️ No se detectó texto válido en los recortes. Fallback OCR imagen completa...")
        preprocessed = ImagePreprocessor.preprocesar(img)
        intentar_ocr(ImagePreprocessor.escalar_imagen(preprocessed), 'fallback_full_image.png')
        patente = img

    # Volver a verificar por si en el fallback aparece algo
        validos = [
            (t, c) for t, c in todos_los_resultados
            if re.fullmatch(r'[A-Z]{3}\d{3}', t) or re.fullmatch(r'[A-Z]{2}\d{3}[A-Z]{2}', t)
        ]
    # --- Combinar resultados, generar combinaciones, filtrar finales ---
    candidatos_validos = []
    solo_letras = []
    solo_numeros = []

    for t, c in todos_los_resultados:
        t = t.strip().upper()
        corregido = PlateTextProcessor.corregir_patente_por_formato(t)
        corregido = PlateTextProcessor.corregir_prefijo_patente(corregido)

        if re.fullmatch(r'[A-Z]{3}\d{3}', corregido) or re.fullmatch(r'[A-Z]{2}\d{3}[A-Z]{2}', corregido):
            candidatos_validos.append((corregido, c))
        elif len(corregido) in (6, 7) and c >= 0.90 and re.fullmatch(r'[A-Z0-9]+', corregido):
            print(f"⚠️ Agregado por confianza alta: {corregido} (conf: {c:.2f})")
            if len(corregido) == 6 and re.fullmatch(r'[A-Z]{3}\d{3}', corregido):
                candidatos_validos.append((corregido, c))
            elif len(corregido) == 7 and re.fullmatch(r'[A-Z]{2}\d{3}[A-Z]{2}', corregido):
                candidatos_validos.append((corregido, c))
            else:
                print(f"⚠️ Descarta por estructura: {corregido}")
        elif re.fullmatch(r'[A-Z]{2,3}', corregido):
            solo_letras.append((corregido, c))
        elif re.fullmatch(r'\d{3,4}', corregido):
            solo_numeros.append((corregido, c))

    combinaciones_extra = []
    for letras, c1 in solo_letras:
        for numeros, c2 in solo_numeros:
            combinado = letras + numeros
            confianza_prom = round((c1 + c2) / 2, 2)
            if re.fullmatch(r'[A-Z]{3}\d{3}', combinado) or re.fullmatch(r'[A-Z]{2}\d{3}[A-Z]{2}', combinado):
                combinado_corregido = PlateTextProcessor.corregir_prefijo_patente(combinado)
                combinaciones_extra.append((combinado_corregido, confianza_prom))
            elif len(combinado) in (6, 7) and confianza_prom >= 0.90:
                combinado_corregido = PlateTextProcessor.corregir_patente_por_formato(combinado)
                combinado_corregido = PlateTextProcessor.corregir_prefijo_patente(combinado_corregido)
                if (len(combinado_corregido) == 6 and re.fullmatch(r'[A-Z]{3}\d{3}', combinado_corregido)) or \
                   (len(combinado_corregido) == 7 and re.fullmatch(r'[A-Z]{2}\d{3}[A-Z]{2}', combinado_corregido)):
                    combinaciones_extra.append((combinado_corregido, confianza_prom))
                else:
                    print(f"⚠️ Descarta combinado por estructura: {combinado_corregido}")

    candidatos_validos.extend(combinaciones_extra)

    # --- Eliminar duplicados exactos ---
    vistos = set()
    candidatos_sin_repetir = []
    for texto, conf in sorted(candidatos_validos, key=lambda x: x[1], reverse=True):
        if texto not in vistos:
            vistos.add(texto)
            candidatos_sin_repetir.append((texto, conf))

    # --- Ajuste prefijo final si es necesario ---
    patentes_finales = []
    vistos_finales = set()
    for texto, conf in candidatos_sin_repetir:
        if re.fullmatch(r'[A-Z]{2}\d{3}[A-Z]{2}', texto) and not texto.startswith('A'):
            corregido = 'A' + texto[1:]
            print(f"🔧 Corregido en filtro final: {texto} → {corregido}")
            texto = corregido
        if texto not in vistos_finales:
            vistos_finales.add(texto)
            patentes_finales.append((texto, conf))

    # --- Mostrar resultados finales ---
    if patentes_finales:
        print(" Patentes válidas detectadas (finales):")
        for i, (pat, conf) in enumerate(patentes_finales, 1):
            print(f"{i}. {pat} (confianza: {conf:.2f})")
        return [p for p, _ in patentes_finales], patente
    else:
        print("⚠️ No se detectó ningún texto válido.")
        return [p for p, _ in patentes_finales], patente

