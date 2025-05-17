import torch
import numpy as np
from app.config import MODELO_PATH, DEVICE
from models.experimental import attempt_load
from utils.general import non_max_suppression
import cv2

# Cargar modelo una sola vez
model = attempt_load(str(MODELO_PATH), device=DEVICE)
model.eval()

def detectar_patente(image_np: np.ndarray):
    h, w = image_np.shape[:2]
    img_resized = cv2.resize(image_np, (640, 640))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().div(255).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    if pred is None or len(pred) == 0:
        raise ValueError("❌ No se detectó ninguna patente.")

    xyxy = pred[0][:4].cpu().numpy()
    x1, y1, x2, y2 = (xyxy * np.array([w/640, h/640, w/640, h/640])).astype(int)
    return image_np[y1:y2, x1:x2]
