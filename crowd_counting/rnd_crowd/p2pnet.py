#!/usr/bin/env python3
"""
P2PNet: Purely Point-Based Crowd Counting and Localization.
Self-contained script for running inference on an image.
Usage:
    python p2pnet.py <image_path> [--threshold 0.5] [--weight weights/p2pnet_SHTechA.pth] [--output overlay.jpg]
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T
from torchvision.models import vgg16_bn

# --- P2PNet Architecture Definition (Self-Contained) ---

class RegressionModel(nn.Module):
    def __init__(self, num_features_in=256, num_anchor_points=4, feature_size=256):
        super(RegressionModel, self).__init__()
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchor_points * 2, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.output(out)
        out = out.permute(0, 2, 3, 1)
        return out.contiguous().view(out.shape[0], -1, 2)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in=256, num_anchor_points=4, num_classes=2, feature_size=256):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchor_points = num_anchor_points
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(feature_size, num_anchor_points * num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))
        out = self.output(out)
        out1 = out.permute(0, 2, 3, 1)
        b, h, w, _ = out1.shape
        out2 = out1.view(b, h, w, self.num_anchor_points, self.num_classes)
        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


def _generate_anchor_points(stride=16, row=2, line=2):
    row_step = stride / row
    line_step = stride / line
    shift_x = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    shift_y = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    return np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()


def _shift(shape, stride, anchor_points):
    shift_x = (np.arange(0, shape[1]) + 0.5) * stride
    shift_y = (np.arange(0, shape[0]) + 0.5) * stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()
    A = anchor_points.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchor_points.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2)))
    return all_anchors.reshape((K * A, 2))


class AnchorPoints(nn.Module):
    def __init__(self, pyramid_levels=None, row=2, line=2):
        super(AnchorPoints, self).__init__()
        self.pyramid_levels = [3] if pyramid_levels is None else pyramid_levels
        self.strides = [2 ** x for x in self.pyramid_levels]
        self.row = row
        self.line = line

    def forward(self, image):
        image_shape = np.array(image.shape[2:])
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        all_anchor_points = np.zeros((0, 2), dtype=np.float32)
        for idx, p in enumerate(self.pyramid_levels):
            anchors = _generate_anchor_points(2 ** p, row=self.row, line=self.line)
            shifted = _shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchor_points = np.append(all_anchor_points, shifted, axis=0)
        all_anchor_points = np.expand_dims(all_anchor_points, axis=0)
        return torch.from_numpy(all_anchor_points.astype(np.float32)).to(image.device)


class Decoder(nn.Module):
    def __init__(self, c3_size=256, c4_size=512, c5_size=512, feature_size=256):
        super(Decoder, self).__init__()
        self.P5_1 = nn.Conv2d(c5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P4_1 = nn.Conv2d(c4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.P3_1 = nn.Conv2d(c3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        c3, c4, c5 = inputs
        p5_x = self.P5_2(self.P5_1(c5))
        p4_x = self.P4_2(self.P4_1(c4) + self.P5_upsampled(self.P5_1(c5)))
        p3_x = self.P3_2(self.P3_1(c3) + self.P4_upsampled(p4_x))
        return [p3_x, p4_x, p5_x]


class BackboneBase_VGG(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        features = list(backbone.features.children())
        self.body1 = nn.Sequential(*features[:13])
        self.body2 = nn.Sequential(*features[13:23])
        self.body3 = nn.Sequential(*features[23:33])
        self.body4 = nn.Sequential(*features[33:43])

    def forward(self, x):
        out = []
        for layer in [self.body1, self.body2, self.body3, self.body4]:
            x = layer(x)
            out.append(x)
        return out


class P2PNetModel(nn.Module):
    def __init__(self, row=2, line=2):
        super().__init__()
        self.backbone = BackboneBase_VGG(vgg16_bn(weights=None))
        self.regression = RegressionModel(num_features_in=256, num_anchor_points=row * line)
        self.classification = ClassificationModel(num_features_in=256, num_classes=2, num_anchor_points=row * line)
        self.anchor_points = AnchorPoints(pyramid_levels=[3], row=row, line=line)
        self.fpn = Decoder(256, 512, 512)

    def forward(self, x):
        features = self.backbone(x)
        features_fpn = self.fpn([features[1], features[2], features[3]])
        regression = self.regression(features_fpn[1]) * 100
        classification = self.classification(features_fpn[1])
        anchors = self.anchor_points(x).repeat(x.shape[0], 1, 1)
        return {'pred_logits': classification, 'pred_points': regression + anchors}


def main():
    parser = argparse.ArgumentParser(description="P2PNet Crowd Counting & Localization")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence score threshold for detections (default: 0.5)")
    parser.add_argument("--weight", default=None,
                        help="Path to pretrained P2PNet checkpoint (.pth)")
    parser.add_argument("--output", default=None, help="Path to save visual overlay")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(json.dumps({"error": f"Image not found: {args.image_path}"}))
        sys.exit(1)

    # Resolve weight path
    weight_path = args.weight
    if not weight_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_p = os.path.join(script_dir, "weights", "p2pnet_SHTechA.pth")
        if os.path.exists(default_p):
            weight_path = default_p
        else:
            print(json.dumps({"error": f"Weight file not found at {default_p}. Specify with --weight."}))
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = P2PNetModel()
    ckpt = torch.load(weight_path, map_location=device)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    img_raw = Image.open(args.image_path).convert('RGB')
    orig_w, orig_h = img_raw.size

    # Round size to multiple of 128 as required by P2PNet
    new_w = max(128, (orig_w // 128) * 128)
    new_h = max(128, (orig_h // 128) * 128)
    img_resized = img_raw.resize((new_w, new_h), Image.Resampling.LANCZOS)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_img = transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor_img)

    scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
    points = outputs['pred_points'][0]

    mask = scores > args.threshold
    detected_points = points[mask].detach().cpu().numpy()
    detected_scores = scores[mask].detach().cpu().numpy()
    count = int(mask.sum().item())

    # Scale points back to original image coordinates
    scale_x = orig_w / float(new_w)
    scale_y = orig_h / float(new_h)

    # Render visual overlay
    orig_bgr = cv2.imread(args.image_path)
    dot_radius = max(2, int(min(orig_w, orig_h) / 250))
    for pt in detected_points:
        px = int(pt[0] * scale_x)
        py = int(pt[1] * scale_y)
        cv2.circle(orig_bgr, (px, py), dot_radius, (0, 0, 255), -1)
        cv2.circle(orig_bgr, (px, py), dot_radius + 1, (255, 255, 255), 1)

    cv2.putText(orig_bgr, f"P2PNet: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    out_path = args.output
    if not out_path:
        stem = Path(args.image_path).stem
        out_dir = Path(args.image_path).parent
        out_path = str(out_dir / f"{stem}_P2PNet_overlay.jpg")

    cv2.imwrite(out_path, orig_bgr)

    res = {
        "method": "P2PNet",
        "image": os.path.basename(args.image_path),
        "count": count,
        "threshold": args.threshold,
        "overlay": out_path
    }
    print(json.dumps(res))

if __name__ == "__main__":
    main()
