from fastapi import FastAPI
from app.routers.plates import router as plates_router

app = FastAPI(
    title="ANPR Plate Detection API",
    description="Sube una imagen y recibe el recorte + texto usando v8.py.",
    version="1.0"
)
app.include_router(plates_router, prefix="/plates", tags=["plates"])

@app.get("/", tags=["root"])
async def root():
    return {"message": "API running con v8.py!"}