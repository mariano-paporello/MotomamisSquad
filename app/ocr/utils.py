import re

def formatear_patente(texto):
    texto = texto.upper().replace(" ", "").replace("\n", "")
    return re.sub(r'[^A-Z0-9]', '', texto)

def es_patente_valida(txt):
    return bool(
        re.match(r'^[A-Z]{3}\d{3}$', txt) or         # ABC123
        re.match(r'^[A-Z]{2}\d{3}[A-Z]{2}$', txt) or # AB123CD
        re.match(r'^[A-Z]{2}\d{4}$', txt) or         # KA0595
        re.match(r'^[A-Z]{3}\d{2}$', txt)            # ABC12
    )
