import torch
import torch.nn as nn
from torchvision.models.video import Swin3D_T_Weights, swin3d_t


class Aux3DEncoder(nn.Module):
    def __init__(self, in_ch: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2)),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiModalSwinClassifier(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.2, flow_weight: float = 1.0, crowd_weight: float = 1.0) -> None:
        super().__init__()
        weights = Swin3D_T_Weights.DEFAULT
        self.rgb_backbone = swin3d_t(weights=weights, progress=True)
        rgb_dim = self.rgb_backbone.head.in_features
        self.rgb_backbone.head = nn.Identity()

        self.flow_encoder = Aux3DEncoder(in_ch=2, out_dim=128)
        self.crowd_encoder = Aux3DEncoder(in_ch=1, out_dim=128)
        self.flow_weight = flow_weight
        self.crowd_weight = crowd_weight

        self.classifier = nn.Sequential(
            nn.Linear(rgb_dim + 128 + 128, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, rgb: torch.Tensor, flow: torch.Tensor, crowd: torch.Tensor) -> torch.Tensor:
        rgb = rgb.permute(0, 2, 1, 3, 4)
        flow = flow.permute(0, 2, 1, 3, 4)
        crowd = crowd.permute(0, 2, 1, 3, 4)

        rgb_feat = self.rgb_backbone(rgb)
        flow_feat = self.flow_encoder(flow) * self.flow_weight
        crowd_feat = self.crowd_encoder(crowd) * self.crowd_weight
        fused = torch.cat([rgb_feat, flow_feat, crowd_feat], dim=1)
        return self.classifier(fused)
