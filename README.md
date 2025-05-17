# MotomamisSquad

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0+-green.svg)
![Uvicorn](https://img.shields.io/badge/Uvicorn-0.23.0+-purple.svg)

## Descripción del Proyecto

> MotomamisSquad es una API desarrollada con FastAPI que utiliza visión por computadora para detectar y reconocer matrículas de vehículos en imágenes. Se integra con modelos de detección de objetos (como YOLOv5) y una librería de OCR (como PaddleOCR) para proporcionar una solución completa de reconocimiento de patentes.

## Funcionalidades Principales

* Detección de matrículas en imágenes.
* Reconocimiento de caracteres de las matrículas (OCR).
* API RESTful para interactuar con el servicio.

## Tecnologías Utilizadas

Principales tecnologías y librerías:

* **Python:** Lenguaje de programación principal.
* **FastAPI:** Framework web moderno y de alto rendimiento para construir APIs.
* **Uvicorn:** Servidor ASGI para ejecutar aplicaciones FastAPI.
* **YOLOv5:** Framework de detección de objetos en tiempo real.
* **PaddleOCR:** Librería de reconocimiento óptico de caracteres.
* **OpenCV (cv2):** Librería para el procesamiento de imágenes.
* **Pydantic:** Para la validación de datos y serialización.

## Requisitos

Lista de los requisitos para ejecutar el proyecto:

* Python 3.10 o superior.
* pip (gestor de paquetes de Python).
* [Cualquier otra dependencia del sistema operativo o software específico].

## Instalación


1.  **Clona el repositorio:**

    ```bash
    git clone <URL_DE_TU_REPOSITORIO>
    cd MotomamisSquad
    ```

2.  **Crea un entorno virtual (recomendado):**

    ```bash
    python -m venv venv
    source venv/bin/activate   # En macOS/Linux
    # venv\Scripts\activate  # En Windows
    ```

3.  **Instala las dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Descarga los modelos necesarios (si aplica para YOLOv5 y PaddleOCR):**

    *SE TIENE QUE COMPLETAR*

## Uso

Instrucciones sobre cómo utilizar la API o la aplicación:

1.  **Ejecuta la API:**

    ```bash
    uvicorn app.main:app --reload
    ```

2.  **Accede a la documentación de la API:**

    Ve a `http://127.0.0.1:8000/docs` en tu navegador para interactuar con la documentación Swagger de la API.

3.  **Ejemplos de endpoints:**

    * `POST /detect`: Envía una imagen para detectar y reconocer la matrícula. Espera un archivo de imagen y devuelve la matrícula detectada (si se encuentra).