from pathlib import Path
import sys

# Obtiene el directorio donde se encuentra el archivo actual (por ejemplo, config.py)
PROJECT_ROOT = Path(__file__).parent.parent
print(PROJECT_ROOT)
YOLOV5_PATH = PROJECT_ROOT / 'yolov5'
print(YOLOV5_PATH)
sys.path.insert(0, str(YOLOV5_PATH))

# Ruta al modelo entrenado
MODELO_PATH = Path(__file__).parent / 'model' / 'best_windows_safe.pt'

# Dispositivo
from utils.torch_utils import select_device
DEVICE = select_device('cpu')