from pydantic import BaseModel
from typing import List

class PlateResponse(BaseModel):
    plate: List[str]  # ← Antes era solo "str"
    image: str        # Base64 de la imagen recortada
