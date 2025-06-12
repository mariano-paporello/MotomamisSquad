import sys
import os
import cv2
import torch
import numpy as np
import re
from pathlib import Path
from paddleocr import PaddleOCR
from scipy.signal import wiener
from skimage import exposure, img_as_ubyte
from skimage.color import rgb2gray

BASE_DIR = Path(__file__).resolve().parent.parent
YOLOV5_DIR = BASE_DIR / 'yolov5'
sys.path.insert(0, str(YOLOV5_DIR))

from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device

ocr_engine = PaddleOCR(use_angle_cls=True, lang='en')
device = select_device('cpu')
model_path = BASE_DIR / 'model' / 'best_windows_safe.pt'
detection_model = attempt_load(str(model_path), device)
detection_model.eval()

class PlateTextProcessor:
    @staticmethod
    def normalize_text(text):
        return re.sub(r'[^A-Z0-9]', '', text.upper().replace(" ", "").replace("\n", ""))

    @staticmethod
    def is_valid_plate(text):
        return bool(re.match(r'^[A-Z]{3}\d{3}$', text) or re.match(r'^[A-Z]{2}\d{3}[A-Z]{2}$', text))

    @staticmethod
    def format_text_like_plate(text):
        text = PlateTextProcessor.normalize_text(text)
        if re.fullmatch(r'[A-Z0-9]{2}[A-Z0-9]{3}[A-Z0-9]{2}', text):
            return text[:2].replace("1", "I").replace("0", "O") + \
                   text[2:5].replace("I", "1").replace("O", "0") + \
                   text[5:].replace("1", "I").replace("0", "O")
        if re.fullmatch(r'[A-Z0-9]{3}[A-Z0-9]{3}', text):
            return text[:3].replace("1", "I").replace("0", "O") + \
                   text[3:].replace("I", "1").replace("O", "0")
        return text

class ImagePreprocessor:
    def __init__(self, image: np.ndarray):
        self.original = image.copy()
        self.image = image.copy()

    def reset(self):
        self.image = self.original.copy()
        return self
    
    def clahe_binarized_scaled(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(gray)
        _, binary = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        scaled = cv2.resize(binary, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        self.image = scaled
        return self


    def enhance_sharpness_and_clahe(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 10.0)
        sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.image = clahe.apply(sharp)
        return self

    def enhance_black_letters(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        self.image = cv2.bitwise_not(dilated)
        return self

    def apply_wiener_filter(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        filtered = wiener(gray)
        self.image = np.uint8(np.clip(filtered, 0, 255))
        return self

    def improve_contrast(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        p2, p98 = np.percentile(gray, (2, 98))
        rescaled = exposure.rescale_intensity(gray, in_range=(p2, p98))
        equalized = exposure.equalize_adapthist(rescaled, clip_limit=0.03)
        self.image = img_as_ubyte(equalized)
        return self

    def preprocess_basic(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(gray)
        _, binary = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.image = cv2.bitwise_not(binary) if np.mean(binary) < 127 else binary
        return self

    def scale_image(self, factor=3.0):
        self.image = cv2.resize(self.image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
        return self

    def crop_by_projections(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
        edges = cv2.filter2D(enhanced, -1, kernel)
        structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        opened = cv2.morphologyEx(edges, cv2.MORPH_OPEN, structure)

        vertical_projection = np.sum(opened, axis=0)
        horizontal_projection = np.sum(opened, axis=1)

        v_threshold = 0.2 * np.max(vertical_projection)
        h_threshold = 0.2 * np.max(horizontal_projection)

        cols = np.where(vertical_projection > v_threshold)[0]
        rows = np.where(horizontal_projection > h_threshold)[0]

        if rows.size and cols.size:
            ymin, ymax = rows[0], rows[-1]
            xmin, xmax = cols[0], cols[-1]
            self.image = self.image[ymin:ymax, xmin:xmax]
        return self

    def get_image(self):
        return self.image

class PlateDetector:
    def __init__(self):
        self.results = []

    def run_ocr(self, image: np.ndarray, save_path: Path) -> tuple[str, float]:
        cv2.imwrite(str(save_path), image)
        ocr_result = ocr_engine.ocr(image, cls=True)
        candidates = []
        if ocr_result:
            for line in ocr_result:
                if not line:
                    continue
                for box in line:
                    if box and len(box) == 2 and isinstance(box[1], tuple):
                        text, confidence = box[1]
                        formatted = PlateTextProcessor.normalize_text(text)
                        if PlateTextProcessor.is_valid_plate(formatted):
                            return formatted, confidence
                        candidates.append((formatted, confidence))
        return max(candidates, key=lambda x: x[1], default=("", 0.0))

    async def detect_from_bytes(self, image_bytes: bytes) -> tuple[str, np.ndarray]:
        output_dir = BASE_DIR / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Leer imagen
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Invalid image data.")

        # Preprocesamiento adaptativo para mejorar detección
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        enhanced_image = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

        # Redimensionar imagen para detector
        h, w = image.shape[:2]
        resized = cv2.resize(enhanced_image, (640, 640))
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)

        # Ejecutar detección
        with torch.no_grad():
            predictions = detection_model(tensor)[0]
            predictions = non_max_suppression(predictions, conf_thres=0.25, iou_thres=0.45)[0]

        # Si hay detección, usar esa región
        if predictions is not None and len(predictions) > 0:
            coords = predictions[0][:4].cpu().numpy()
            x1, y1, x2, y2 = (coords * np.array([w/640, h/640, w/640, h/640])).astype(int)

            cropped_plate = image[y1:y2, x1:x2]
            cv2.imwrite(str(output_dir / "debug_raw_plate.png"), cropped_plate)
            raw_ocr_result = self.run_ocr(cropped_plate, output_dir / "raw_ocr.png")
            self.results.append(raw_ocr_result)
            print(f"Raw OCR: {raw_ocr_result[0]} (conf: {raw_ocr_result[1]:.2f})")

            processor = ImagePreprocessor(cropped_plate)
            transformations = [
                (processor.reset().scale_image().get_image(), 'scaled.png'),
                (processor.reset().apply_wiener_filter().get_image(), 'wiener.png'),
                (processor.reset().enhance_sharpness_and_clahe().get_image(), 'sharp_clahe.png'),
                (processor.reset().enhance_black_letters().get_image(), 'black_letters.png'),
                (processor.reset().improve_contrast().get_image(), 'contrast.png'),
                (processor.reset().preprocess_basic().scale_image().get_image(), 'preproc.png'),
            ]

            for img, name in transformations:
                result = self.run_ocr(img, output_dir / name)
                self.results.append(result)

            valid = [r for r in self.results if PlateTextProcessor.is_valid_plate(r[0])]
            if valid:
                best, conf = max(valid, key=lambda x: x[1])
                return f"License plate: {best} (conf: {conf:.2f})", cropped_plate

        # Fallback: usar OCR en imagen completa si no hubo detección
        print("No plate detected by model. Trying full image OCR fallback...")
        preprocessed = ImagePreprocessor(image).enhance_sharpness_and_clahe().scale_image().get_image()
        full_ocr_result = self.run_ocr(preprocessed, output_dir / "fallback_full_ocr.png")
        if PlateTextProcessor.is_valid_plate(full_ocr_result[0]):
            return f"Fallback plate: {full_ocr_result[0]} (conf: {full_ocr_result[1]:.2f})", image

        print("No valid license plate detected.")
        return "No license plate detected.", image