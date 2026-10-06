import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any
import pandas as pd

class QualityAnalyzer:
    """
    Extracts essential CV image quality metrics:
    - Brightness
    - Contrast
    - Sharpness / Blur (Laplacian variance)
    - Noise estimate (standard deviation of high-frequency components)
    - Resolution / Dimensions
    """

    @staticmethod
    def analyze_image(image: Image.Image) -> Dict[str, float]:
        # Convert to numpy and grayscale
        img_np = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        h, w = gray.shape

        # Brightness (mean pixel intensity)
        brightness = float(np.mean(gray))

        # Contrast (standard deviation of pixel intensity)
        contrast = float(np.std(gray))

        # Sharpness / Blur metric (variance of Laplacian)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = float(laplacian.var())

        # Noise estimation using median filter difference
        blur_med = cv2.medianBlur(gray, 3)
        noise = float(np.std(gray.astype(np.float64) - blur_med.astype(np.float64)))

        # Aspect ratio
        aspect_ratio = float(w / h) if h > 0 else 1.0

        return {
            "width": float(w),
            "height": float(h),
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "noise": noise,
            "aspect_ratio": aspect_ratio
        }

    @classmethod
    def analyze_dataset(cls, images: List[Image.Image]) -> pd.DataFrame:
        records = [cls.analyze_image(img) for img in images]
        return pd.DataFrame(records)
