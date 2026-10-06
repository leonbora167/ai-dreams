import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Dict, Any

class ImagePerturbation:
    """
    Applies controlled distortions simulating real-world camera degradation and drift:
    - Gaussian Blur / Out-of-focus
    - Motion Blur
    - Sensor Noise (Gaussian / Poisson)
    - Exposure / Brightness shift
    - Contrast degradation
    - JPEG compression artifacts
    - Color shifts
    """

    @staticmethod
    def apply_gaussian_blur(image: Image.Image, radius: float = 3.0) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    @staticmethod
    def apply_brightness_shift(image: Image.Image, factor: float = 0.4) -> Image.Image:
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    @staticmethod
    def apply_contrast_shift(image: Image.Image, factor: float = 0.4) -> Image.Image:
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @staticmethod
    def apply_sensor_noise(image: Image.Image, sigma: float = 25.0) -> Image.Image:
        img_np = np.array(image.convert("RGB")).astype(np.float32)
        noise = np.random.normal(0, sigma, img_np.shape)
        noisy_img = np.clip(img_np + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    @staticmethod
    def apply_jpeg_compression(image: Image.Image, quality: int = 15) -> Image.Image:
        import io
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    @classmethod
    def apply_camera_degradation(cls, image: Image.Image) -> Image.Image:
        """Simulates dirty lens + low-light sensor noise + blur."""
        img = cls.apply_gaussian_blur(image, radius=2.5)
        img = cls.apply_brightness_shift(img, factor=0.6)
        img = cls.apply_sensor_noise(img, sigma=20.0)
        return img
