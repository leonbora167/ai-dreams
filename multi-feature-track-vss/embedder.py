import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoProcessor, AutoModel
from huggingface_hub import hf_hub_download
import torchreid


MSMT17_REID_REPO = "kaiyangzhou/osnet"
MSMT17_REID_WEIGHT = (
    "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
)


class LocalEmbedder:
    def __init__(
        self,
        device: str | None = None,
        siglip_id: str = "google/siglip-base-patch16-384"
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # 1. Initialize OSNet-AIN x1.0 with true MSMT17 person Re-ID weights (512-d)
        try:
            ckpt_path = hf_hub_download(repo_id=MSMT17_REID_REPO, filename=MSMT17_REID_WEIGHT)
            self.osnet = torchreid.models.build_model(
                name="osnet_ain_x1_0",
                num_classes=1,
                pretrained=False
            )
            torchreid.utils.load_pretrained_weights(self.osnet, ckpt_path)
            print("Loaded MSMT17 Re-ID weights (osnet_ain_x1_0) for body Re-ID.")
        except Exception as e:
            print(f"Warning: Could not load MSMT17 Re-ID weights ({e}); falling back to osnet_x0_5.")
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

        # 2. Initialize SigLIP (384px resolution by default for small objects like mics)
        self.siglip_id = siglip_id
        self.siglip_processor = AutoProcessor.from_pretrained(siglip_id)
        self.siglip_model = AutoModel.from_pretrained(siglip_id)
        self.siglip_model = self.siglip_model.to(self.device)
        self.siglip_model.eval()
        self.siglip_dim = int(getattr(self.siglip_model.config.text_config, "hidden_size", 768))

    def extract_body_reid(self, crop: np.ndarray) -> list[float]:
        """
        Extracts a 512-d L2-normalized person Re-ID vector using OSNet-AIN (MSMT17).
        """
        if crop is None or crop.size == 0:
            return [0.0] * 512

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = self.osnet_transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.osnet(tensor)
            feat = feat / torch.clamp(feat.norm(p=2, dim=1, keepdim=True), min=1e-12)

        return feat[0].cpu().numpy().tolist()

    def extract_multimodal_image(self, crop: np.ndarray) -> list[float]:
        """
        Extracts an L2-normalized SigLIP visual embedding (768-d for base-384).
        Works on both person crops and full auditorium scene frames.
        """
        if crop is None or crop.size == 0:
            return [0.0] * self.siglip_dim

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
        Extracts an L2-normalized SigLIP text embedding.
        """
        if not text or not text.strip():
            return [0.0] * self.siglip_dim

        inputs = self.siglip_processor(text=[text.strip()], padding="max_length", return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.siglip_model.get_text_features(**inputs)
            feat = out.pooler_output if hasattr(out, "pooler_output") else out
            feat = feat / torch.clamp(feat.norm(p=2, dim=-1, keepdim=True), min=1e-12)

        return feat[0].cpu().numpy().tolist()
