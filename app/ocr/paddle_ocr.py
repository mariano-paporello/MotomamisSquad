from paddleocr import PaddleOCR
from .utils import formatear_patente, es_patente_valida

ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

def ocr_paddle(img):
    resultado = ocr.ocr(img, cls=True)
    posibles = []
    if resultado:
        for linea in resultado:
            if linea:
                for bbox in linea:
                    if bbox and len(bbox) == 2 and isinstance(bbox[1], tuple):
                        txt, conf = bbox[1]
                        if conf > 0.5:
                            texto_limpio = formatear_patente(txt)
                            if es_patente_valida(texto_limpio):
                                return texto_limpio
                            posibles.append(texto_limpio)
    return ""
