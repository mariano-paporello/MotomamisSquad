from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.plates import router as plates_router

app = FastAPI(
    title="API de Detección de Matrículas",
    description="Sube una imagen y recibe el recorte de la matrícula con el texto detectado."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(plates_router, prefix="/plates", tags=["plates"])

@app.get("/", tags=["health"])
async def root():
    return {"message": "API de detección de matrículas en línea"}
