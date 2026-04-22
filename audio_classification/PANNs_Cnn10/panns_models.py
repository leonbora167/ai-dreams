from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_layer(layer: nn.Module) -> None:
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn: nn.BatchNorm2d | nn.BatchNorm1d) -> None:
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, x: torch.Tensor, pool_size: tuple[int, int] = (2, 2)) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=pool_size)
        return x


class _PANNsBase(nn.Module):
    def __init__(
        self,
        channels: list[int],
        embedding_dim: int,
        num_classes: int = 2,
        mel_bins: int = 64,
    ) -> None:
        super().__init__()
        self.bn0 = nn.BatchNorm2d(mel_bins)
        self.blocks = nn.ModuleList(
            [ConvBlock(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.fc1 = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.fc_out = nn.Linear(embedding_dim, num_classes, bias=True)

        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, :1, :, :]
        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        for block in self.blocks:
            x = block(x, pool_size=(2, 2))

        x = torch.mean(x, dim=3)
        x1, _ = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu_(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        return self.fc_out(x)


class Cnn10(_PANNsBase):
    def __init__(self, num_classes: int = 2, mel_bins: int = 64) -> None:
        super().__init__(
            channels=[1, 64, 128, 256, 512],
            embedding_dim=512,
            num_classes=num_classes,
            mel_bins=mel_bins,
        )


class Cnn14(_PANNsBase):
    def __init__(self, num_classes: int = 2, mel_bins: int = 64) -> None:
        super().__init__(
            channels=[1, 64, 128, 256, 512, 1024, 2048],
            embedding_dim=2048,
            num_classes=num_classes,
            mel_bins=mel_bins,
        )
