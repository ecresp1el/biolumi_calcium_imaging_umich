import cv2
import numpy as np

def process_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    processed = cv2.GaussianBlur(image, (5, 5), 0)
    return processed
