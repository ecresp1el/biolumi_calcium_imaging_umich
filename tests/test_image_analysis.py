import cv2
import numpy as np
from BL_CalciumAnalysis.image_analysis_methods_umich import process_image

def test_process_image():
    img = np.ones((100, 100), dtype=np.uint8) * 255
    processed = process_image(img)
    assert processed.shape == img.shape
