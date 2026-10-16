"""FastFlowNet architecture implemented with pure PyTorch correlation (no custom CUDA extensions required).

Reference:
- Paper: "FastFlowNet: A Lightweight Network for Fast Optical Flow Estimation" (ICRA 2021)
- Weights: weights/fastflownet_ft_mix.pth (~5.48 MB)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


def convrelu(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1, groups: int = 1) -> nn.Sequential:
    """Construct a Conv2d + LeakyReLU layer."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
    )


def deconv(in_planes: int, out_planes: int) -> nn.ConvTranspose2d:
    """Construct a 2x spatial upsampling ConvTranspose2d layer."""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True)


class PureCorrelation(nn.Module):
    """Pure PyTorch spatial correlation layer over a local displacement neighborhood."""

    def __init__(self, max_displacement: int = 4):
        super().__init__()
        self.max_displacement = max_displacement

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        pad = self.max_displacement
        y_pad = F.pad(y, (pad, pad, pad, pad), mode="constant", value=0.0)
        out_list = []
        for dy in range(2 * pad + 1):
            for dx in range(2 * pad + 1):
                y_slice = y_pad[:, :, dy : dy + h, dx : dx + w]
                out_list.append(torch.mean(x * y_slice, dim=1, keepdim=True))
        return torch.cat(out_list, dim=1)


class Decoder(nn.Module):
    """FastFlowNet flow decoder module."""

    def __init__(self, in_channels: int, groups: int = 3):
        super().__init__()
        self.groups = groups
        self.conv1 = convrelu(in_channels, 96, 3, 1)
        self.conv2 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv3 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv4 = convrelu(96, 96, 3, 1, groups=groups)
        self.conv5 = convrelu(96, 64, 3, 1)
        self.conv6 = convrelu(64, 32, 3, 1)
        self.conv7 = nn.Conv2d(32, 2, 3, 1, 1)

    def channel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        channels_per_group = c // self.groups
        return x.view(b, self.groups, channels_per_group, h, w).transpose(1, 2).contiguous().view(b, -1, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.channel_shuffle(self.conv2(out))
        out = self.channel_shuffle(self.conv3(out))
        out = self.channel_shuffle(self.conv4(out))
        return self.conv7(self.conv6(self.conv5(out)))


class FastFlowNet(nn.Module):
    """FastFlowNet model with pyramid feature encoder, correlation volume, and flow decoders."""

    def __init__(self, groups: int = 3):
        super().__init__()
        self.groups = groups
        self.pconv1_1 = convrelu(3, 16, 3, 2)
        self.pconv1_2 = convrelu(16, 16, 3, 1)
        self.pconv2_1 = convrelu(16, 32, 3, 2)
        self.pconv2_2 = convrelu(32, 32, 3, 1)
        self.pconv2_3 = convrelu(32, 32, 3, 1)
        self.pconv3_1 = convrelu(32, 64, 3, 2)
        self.pconv3_2 = convrelu(64, 64, 3, 1)
        self.pconv3_3 = convrelu(64, 64, 3, 1)

        self.corr = PureCorrelation(max_displacement=4)
        self.register_buffer(
            "index",
            torch.tensor(
                [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 24, 26, 28, 29, 30, 31, 32, 33, 34,
                 36, 38, 39, 40, 41, 42, 44, 46, 47, 48, 49, 50, 51, 52, 54, 56, 57, 58, 59, 60, 62,
                 64, 66, 68, 70, 72, 74, 76, 78, 80],
                dtype=torch.long
            ),
            persistent=False
        )

        self.rconv2 = convrelu(32, 32, 3, 1)
        self.rconv3 = convrelu(64, 32, 3, 1)
        self.rconv4 = convrelu(64, 32, 3, 1)
        self.rconv5 = convrelu(64, 32, 3, 1)
        self.rconv6 = convrelu(64, 32, 3, 1)

        self.up3 = deconv(2, 2)
        self.up4 = deconv(2, 2)
        self.up5 = deconv(2, 2)
        self.up6 = deconv(2, 2)

        self.decoder2 = Decoder(87, groups)
        self.decoder3 = Decoder(87, groups)
        self.decoder4 = Decoder(87, groups)
        self.decoder5 = Decoder(87, groups)
        self.decoder6 = Decoder(87, groups)

    def warp(self, x: torch.Tensor, flo: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        xx = torch.arange(0, w, device=x.device).view(1, -1).repeat(h, 1)
        yy = torch.arange(0, h, device=x.device).view(-1, 1).repeat(1, w)
        grid = torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(b, 1, 1, 1).float()
        vgrid = grid + flo
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(w - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(h - 1, 1) - 1.0
        return F.grid_sample(x, vgrid.permute(0, 2, 3, 1), mode="bilinear", align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img1 = x[:, :3, :, :]
        img2 = x[:, 3:6, :, :]

        f11 = self.pconv1_2(self.pconv1_1(img1))
        f21 = self.pconv1_2(self.pconv1_1(img2))
        f12 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f11)))
        f22 = self.pconv2_3(self.pconv2_2(self.pconv2_1(f21)))
        f13 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f12)))
        f23 = self.pconv3_3(self.pconv3_2(self.pconv3_1(f22)))

        f14 = F.avg_pool2d(f13, kernel_size=(2, 2), stride=(2, 2))
        f24 = F.avg_pool2d(f23, kernel_size=(2, 2), stride=(2, 2))
        f15 = F.avg_pool2d(f14, kernel_size=(2, 2), stride=(2, 2))
        f25 = F.avg_pool2d(f24, kernel_size=(2, 2), stride=(2, 2))
        f16 = F.avg_pool2d(f15, kernel_size=(2, 2), stride=(2, 2))
        f26 = F.avg_pool2d(f25, kernel_size=(2, 2), stride=(2, 2))

        flow7_up = torch.zeros(f16.size(0), 2, f16.size(2), f16.size(3), device=x.device)
        cv6 = torch.index_select(self.corr(f16, f26), dim=1, index=self.index)
        flow6 = self.decoder6(torch.cat([cv6, self.rconv6(f16), flow7_up], 1))

        flow6_up = self.up6(flow6)
        f25_w = self.warp(f25, flow6_up * 0.625)
        cv5 = torch.index_select(self.corr(f15, f25_w), dim=1, index=self.index)
        flow5 = self.decoder5(torch.cat([cv5, self.rconv5(f15), flow6_up], 1)) + flow6_up

        flow5_up = self.up5(flow5)
        f24_w = self.warp(f24, flow5_up * 1.25)
        cv4 = torch.index_select(self.corr(f14, f24_w), dim=1, index=self.index)
        flow4 = self.decoder4(torch.cat([cv4, self.rconv4(f14), flow5_up], 1)) + flow5_up

        flow4_up = self.up4(flow4)
        f23_w = self.warp(f23, flow4_up * 2.5)
        cv3 = torch.index_select(self.corr(f13, f23_w), dim=1, index=self.index)
        flow3 = self.decoder3(torch.cat([cv3, self.rconv3(f13), flow4_up], 1)) + flow4_up

        flow3_up = self.up3(flow3)
        f22_w = self.warp(f22, flow3_up * 5.0)
        cv2 = torch.index_select(self.corr(f12, f22_w), dim=1, index=self.index)
        flow2 = self.decoder2(torch.cat([cv2, self.rconv2(f12), flow3_up], 1)) + flow3_up

        return flow2


def load_fastflownet(weights_path: str = "weights/fastflownet_ft_mix.pth", device: str = None) -> FastFlowNet:
    """Load FastFlowNet model with pretrained checkpoint onto specified device."""
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = FastFlowNet().to(dev).eval()
    assert os.path.exists(weights_path), f"FastFlowNet weights not found at {weights_path}"
    state = torch.load(weights_path, map_location=dev)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model

