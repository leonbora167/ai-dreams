#!/usr/bin/env python3
"""
APGCC: Improving Point-based Crowd Counting and Localization Based on Auxiliary Point Guidance.
Self-contained script for running inference on an image.
Usage:
    python apgcc.py <image_path> [--threshold 0.5] [--weight weights/apgcc_SHHA_best.pth] [--output overlay.jpg]
"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T
from torchvision.models import vgg16_bn

# --- APGCC Self-Contained Model Definitions ---

def make_coord(shape, ranges=None, flatten=True):
    coord_seqs = []
    for i, n in enumerate(shape):
        v0, v1 = (-1, 1) if ranges is None else ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs, indexing='ij'), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class SpatialEncoding(nn.Module):
    def __init__(self, in_dim=2, out_dim=32, sigma=6.0, cat_input=True, require_grad=False):
        super().__init__()
        self.require_grad = require_grad
        self.cat_input = cat_input
        emb = torch.randn(out_dim // 2, in_dim) * sigma
        if require_grad:
            self.emb = nn.Parameter(emb, requires_grad=True)
        else:
            self.register_buffer('emb', emb)

    def forward(self, x):
        emb = self.emb.to(x.device)
        y = torch.matmul(x, emb.T)
        if self.cat_input:
            return torch.cat([x, torch.sin(y), torch.cos(y)], dim=-1)
        else:
            return torch.cat([torch.sin(y), torch.cos(y)], dim=-1)


def ifi_feat(res, size, stride=1, local=False):
    bs, hh, ww = res.shape[0], res.shape[-2], res.shape[-1]
    h, w = size
    device = res.device
    coords = (make_coord((h, w)).to(device).flip(-1) + 1) / 2
    coords = coords.unsqueeze(0).expand(bs, *coords.shape)
    coords = (coords * 2 - 1).flip(-1)

    feat_coords = make_coord((hh, ww), flatten=False).to(device).permute(2, 0, 1).unsqueeze(0).expand(bs, 2, hh, ww)

    if local:
        vx_list = [-1, 1]
        vy_list = [-1, 1]
        eps_shift = 1e-6
        rel_coord_list = []
        q_feat_list = []
        area_list = []
    else:
        vx_list, vy_list, eps_shift = [0], [0], 0

    rx = stride / h
    ry = stride / w

    for vx in vx_list:
        for vy in vy_list:
            coords_ = coords.clone()
            coords_[:, :, 0] += vx * rx + eps_shift
            coords_[:, :, 1] += vy * ry + eps_shift
            coords_.clamp_(-1 + 1e-6, 1 - 1e-6)
            q_feat = F.grid_sample(res, coords_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
            q_coord = F.grid_sample(feat_coords, coords_.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
            rel_coord = coords - q_coord
            rel_coord[:, :, 0] *= hh
            rel_coord[:, :, 1] *= ww
            area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
            rel_coord_list.append(rel_coord)
            q_feat_list.append(q_feat)
            area_list.append(area + 1e-9)

    return rel_coord_list, q_feat_list, area_list


class ifi_simfpn(nn.Module):
    def __init__(self, pos_dim=32, num_anchor_points=4, num_classes=2, stride=1,
                 head_layers=None, feat_num=2, feat_dim=64):
        super().__init__()
        if head_layers is None:
            head_layers = [1024, 512, 256, 256]
        self.stride = stride
        self.feat_num = feat_num
        self.num_anchor_points = num_anchor_points
        self.num_classes = num_classes

        # Spatial encodings for levels
        self.pos1 = SpatialEncoding(2, pos_dim, require_grad=True)
        self.pos2 = SpatialEncoding(2, pos_dim, require_grad=True)
        effective_pos_dim = pos_dim + 2

        in_dim = feat_num * (feat_dim + effective_pos_dim)

        conf_layers = []
        off_layers = []
        for ct, hf in enumerate(head_layers):
            s_dim = in_dim if ct == 0 else head_layers[ct - 1]
            conf_layers.extend([nn.Conv1d(s_dim, hf, 1), nn.BatchNorm1d(hf), nn.ReLU()])
            off_layers.extend([nn.Conv1d(s_dim, hf, 1), nn.BatchNorm1d(hf), nn.ReLU()])

        conf_layers.extend([nn.Conv1d(head_layers[-1], num_anchor_points * num_classes, 1), nn.ReLU()])
        off_layers.append(nn.Conv1d(head_layers[-1], num_anchor_points * 2, 1))

        self.confidence_head = nn.Sequential(*conf_layers)
        self.offset_head = nn.Sequential(*off_layers)

    def forward(self, x, size, level=0, after_cat=False):
        h, w = size
        if not after_cat:
            rel_coord_list, q_feat_list, area_list = ifi_feat(x, [h, w], local=True, stride=self.stride)
            total_area = torch.stack(area_list).sum(dim=0)
            context_list = []
            pe_module = getattr(self, f"pos{level}")
            for rel_coord, q_feat, area in zip(rel_coord_list, q_feat_list, area_list):
                pe_coord = pe_module(rel_coord)
                context_list.append(torch.cat([pe_coord, q_feat], dim=-1))

            area_list[0], area_list[3] = area_list[3], area_list[0]
            area_list[1], area_list[2] = area_list[2], area_list[1]
            out = 0
            for conte, area in zip(context_list, area_list):
                out = out + conte * ((area / total_area).unsqueeze(-1))
            return out
        else:
            offset = self.offset_head(x).view(x.shape[0], -1, h, w).permute(0, 2, 3, 1)
            b, h_, w_, _ = offset.shape
            offset = offset.view(b, h_, w_, self.num_anchor_points, 2).contiguous().view(b, -1, 2)

            conf = self.confidence_head(x).view(x.shape[0], -1, h, w).permute(0, 2, 3, 1)
            conf = conf.view(b, h_, w_, self.num_anchor_points, self.num_classes).contiguous().view(b, -1, self.num_classes)
            return offset, conf


class ASPP(nn.Module):
    def __init__(self, in_planes=512, inner_planes=64, dilations=(2, 4, 8)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_planes, inner_planes, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_planes, inner_planes, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_planes, inner_planes, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_planes, inner_planes, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.ReLU(inplace=True)
        )
        self.out_planes = (len(dilations) + 2) * inner_planes

    def get_outplanes(self):
        return self.out_planes

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.interpolate(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        return torch.cat((feat1, feat2, feat3, feat4, feat5), 1)


class APGCCAnchorPoints(nn.Module):
    def __init__(self, stride=8, row=2, line=2):
        super().__init__()
        self.stride = stride
        self.row = row
        self.line = line

    def forward(self, image):
        img_shape = np.array(image.shape[2:])
        out_shape = (img_shape + self.stride - 1) // self.stride
        row_step = self.stride / self.row
        line_step = self.stride / self.line

        shift_x = (np.arange(1, self.line + 1) - 0.5) * line_step - self.stride / 2
        shift_y = (np.arange(1, self.row + 1) - 0.5) * row_step - self.stride / 2
        shift_x, shift_y = np.meshgrid(shift_x, shift_y, indexing='ij')
        base_anchors = np.vstack((shift_x.ravel(), shift_y.ravel())).transpose()

        grid_x = (np.arange(0, out_shape[1]) + 0.5) * self.stride
        grid_y = (np.arange(0, out_shape[0]) + 0.5) * self.stride
        grid_x, grid_y = np.meshgrid(grid_x, grid_y, indexing='xy')
        shifts = np.vstack((grid_x.ravel(), grid_y.ravel())).transpose()

        A = base_anchors.shape[0]
        K = shifts.shape[0]
        all_anchors = (base_anchors.reshape((1, A, 2)) + shifts.reshape((1, K, 2)).transpose((1, 0, 2))).reshape((K * A, 2))
        all_anchors = np.expand_dims(all_anchors, axis=0)
        return torch.from_numpy(all_anchors.astype(np.float32)).to(image.device)


class APGCCDecoder(nn.Module):
    def __init__(self, inner_planes=64, row=2, line=2):
        super().__init__()
        self.enc3 = nn.Sequential(nn.Conv2d(512, inner_planes, kernel_size=1), nn.BatchNorm2d(inner_planes), nn.ReLU(inplace=True))
        self.aspp = ASPP(512, inner_planes=inner_planes)
        self.head = nn.Sequential(
            nn.Conv2d(inner_planes * 5, inner_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inner_planes),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        self.ifi = ifi_simfpn(pos_dim=32, num_anchor_points=row * line, num_classes=2, stride=1,
                              head_layers=[1024, 512, 256, 256], feat_num=2, feat_dim=inner_planes)
        self.anchor_points = APGCCAnchorPoints(stride=8, row=row, line=line)

    def forward(self, samples, features):
        ht, wt = features[2].shape[-2], features[2].shape[-1]
        batch_size = features[0].shape[0]

        x3 = self.enc3(features[2])
        x4 = self.head(self.aspp(features[3]))

        context = [
            self.ifi(x3, size=[ht, wt], level=1),
            self.ifi(x4, size=[ht, wt], level=2)
        ]
        context = torch.cat(context, dim=-1).permute(0, 2, 1)
        offset, confidence = self.ifi(context, size=[ht, wt], after_cat=True)

        offset = offset * 100
        anchors = self.anchor_points(samples).repeat(batch_size, 1, 1)
        output_coord = offset + anchors
        return {'pred_logits': confidence, 'pred_points': output_coord}


class APGCCEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = vgg16_bn(weights=None)
        features = list(backbone.features.children())
        self.body1 = nn.Sequential(*features[:13])
        self.body2 = nn.Sequential(*features[13:23])
        self.body3 = nn.Sequential(*features[23:33])
        self.body4 = nn.Sequential(*features[33:43])

    def forward(self, x):
        out = []
        for body in [self.body1, self.body2, self.body3, self.body4]:
            x = body(x)
            out.append(x)
        return out


class APGCCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = APGCCEncoder()
        self.decoder = APGCCDecoder()

    def forward(self, x):
        feats = self.encoder(x)
        return self.decoder(x, feats)


def main():
    parser = argparse.ArgumentParser(description="APGCC Crowd Counting & Localization")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence score threshold for detections (default: 0.5)")
    parser.add_argument("--weight", default=None,
                        help="Path to pretrained APGCC checkpoint (.pth)")
    parser.add_argument("--output", default=None, help="Path to save visual overlay")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(json.dumps({"error": f"Image not found: {args.image_path}"}))
        sys.exit(1)

    weight_path = args.weight
    if not weight_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_p = os.path.join(script_dir, "weights", "apgcc_SHHA_best.pth")
        if os.path.exists(default_p):
            weight_path = default_p
        else:
            print(json.dumps({"error": f"Weight file not found at {default_p}. Specify with --weight."}))
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = APGCCModel()
    ckpt = torch.load(weight_path, map_location=device)
    state = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    img_raw = Image.open(args.image_path).convert('RGB')
    orig_w, orig_h = img_raw.size

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
    count = int(mask.sum().item())

    scale_x = orig_w / float(new_w)
    scale_y = orig_h / float(new_h)

    orig_bgr = cv2.imread(args.image_path)
    dot_radius = max(2, int(min(orig_w, orig_h) / 250))
    for pt in detected_points:
        px = int(pt[0] * scale_x)
        py = int(pt[1] * scale_y)
        cv2.circle(orig_bgr, (px, py), dot_radius, (0, 255, 0), -1)
        cv2.circle(orig_bgr, (px, py), dot_radius + 1, (255, 255, 255), 1)

    cv2.putText(orig_bgr, f"APGCC: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    out_path = args.output
    if not out_path:
        stem = Path(args.image_path).stem
        out_dir = Path(args.image_path).parent
        out_path = str(out_dir / f"{stem}_APGCC_overlay.jpg")

    cv2.imwrite(out_path, orig_bgr)

    res = {
        "method": "APGCC",
        "image": os.path.basename(args.image_path),
        "count": count,
        "threshold": args.threshold,
        "overlay": out_path
    }
    print(json.dumps(res))

if __name__ == "__main__":
    main()
