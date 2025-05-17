import cv2
import numpy as np

def preprocesar(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)
    _, binarizada = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binarizada) < 127:
        binarizada = cv2.bitwise_not(binarizada)
    return binarizada

def escalar_imagen(img, factor=3.0):
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)

def warp_perspective(img):
    pts1 = np.float32([[0,0], [img.shape[1],0], [0,img.shape[0]], [img.shape[1], img.shape[0]]])
    pts2 = np.float32([[0,0], [200,0], [0,50], [200,50]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (200,50))
