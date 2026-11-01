import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoProcessor, AutoModel
import torchreid

class LocalEmbedder:
    def __init__(self, device: str | None = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 1. Initialize OSNet x0_5 for 512-d body Re-ID
        self.osnet = torchreid.models.build_model(
            name="osnet_x0_5",
            num_classes=1000,
            pretrained=True
        )
        self.osnet = self.osnet.to(self.device)
        self.osnet.eval()

        # OSNet standard image transformation (H=256, W=128)
        self.osnet_transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 2. Initialize SigLIP for 768-d multimodal image/text retrieval
        siglip_id = "google/siglip-base-patch16-224"
        self.siglip_processor = AutoProcessor.from_pretrained(siglip_id)
        self.siglip_model = AutoModel.from_pretrained(siglip_id)
        self.siglip_model = self.siglip_model.to(self.device)
        self.siglip_model.eval()

    def extract_body_reid(self, crop: np.ndarray) -> list[float]:
        """
        Extracts a 512-d L2-normalized feature vector using OSNet.
        """
        if crop is None or crop.size == 0:
            return [0.0] * 512

        # Convert BGR (OpenCV) to RGB (PIL)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = self.osnet_transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.osnet(tensor)
            feat = feat / torch.clamp(feat.norm(p=2, dim=1, keepdim=True), min=1e-12)

        return feat[0].cpu().numpy().tolist()

    def extract_multimodal_image(self, crop: np.ndarray) -> list[float]:
        """
        Extracts a 768-d L2-normalized image embedding using SigLIP.
        """
        if crop is None or crop.size == 0:
            return [0.0] * 768

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        
        inputs = self.siglip_processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.siglip_model.get_image_features(**inputs)
            feat = out.pooler_output if hasattr(out, "pooler_output") else out
            feat = feat / torch.clamp(feat.norm(p=2, dim=-1, keepdim=True), min=1e-12)

        return feat[0].cpu().numpy().tolist()

    def extract_multimodal_text(self, text: str) -> list[float]:
        """
        Extracts a 768-d L2-normalized text embedding using SigLIP.
        """
        if not text or not text.strip():
            return [0.0] * 768

        inputs = self.siglip_processor(text=[text.strip()], padding="max_length", return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.siglip_model.get_text_features(**inputs)
            feat = out.pooler_output if hasattr(out, "pooler_output") else out
            feat = feat / torch.clamp(feat.norm(p=2, dim=-1, keepdim=True), min=1e-12)

        return feat[0].cpu().numpy().tolist()

