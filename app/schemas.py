from pydantic import BaseModel

class PlateResponse(BaseModel):
    plate: str          # Texto de la matrícula detectada
    image: str          # Imagen recortada en base64 (data URI)