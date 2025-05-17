import cv2
from app.detection.yolo_detector import detectar_patente
from app.detection.image_preprocessing import (
    escalar_imagen, preprocesar, warp_perspective
)
from app.ocr.paddle_ocr import ocr_paddle
from app.ocr.utils import es_patente_valida

def process_image(image_np):
    patente_recortada = detectar_patente(image_np)
    intentos = []

    variantes = [
        escalar_imagen(patente_recortada),
        escalar_imagen(cv2.bitwise_not(patente_recortada)),
        escalar_imagen(preprocesar(patente_recortada)),
        escalar_imagen(warp_perspective(patente_recortada)),
        escalar_imagen(cv2.bitwise_not(warp_perspective(patente_recortada))),
        cv2.resize(warp_perspective(patente_recortada), None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    ]

    for variante in variantes:
        texto = ocr_paddle(variante)
        if es_patente_valida(texto):
            return texto, patente_recortada
        intentos.append(texto)

    return "", patente_recortada
