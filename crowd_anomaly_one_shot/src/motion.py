"""Motion estimation module for Phase 1A.

Units note:
Flow vectors (dx, dy) and speed represent displacements in pixels per analysis-frame interval
(i.e., per 1/ANALYSIS_FPS seconds).
"""

import cv2
import numpy as np
import torch


def flow_dis(prev_frame: np.ndarray, next_frame: np.ndarray, preset: str = "medium") -> np.ndarray:
    """Compute dense optical flow between two consecutive BGR frames using OpenCV DIS, returning (H, W, 2) float32."""
    assert prev_frame.shape == next_frame.shape, f"Frame shape mismatch: {prev_frame.shape} vs {next_frame.shape}"
    assert prev_frame.ndim == 3 and prev_frame.shape[2] == 3, f"Expected (H, W, 3) BGR, got {prev_frame.shape}"
    assert prev_frame.dtype == np.uint8 and next_frame.dtype == np.uint8, "Frames must be uint8"

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    preset_flag = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    if preset.lower() == "fast":
        preset_flag = cv2.DISOPTICAL_FLOW_PRESET_FAST
    elif preset.lower() == "ultrafast":
        preset_flag = cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST

    dis = cv2.DISOpticalFlow_create(preset_flag)
    flow = dis.calc(prev_gray, next_gray, None)
    assert flow.ndim == 3 and flow.shape[2] == 2, f"Expected (H, W, 2) flow, got {flow.shape}"
    return flow.astype(np.float32)


def flow_frame_diff(prev_frame: np.ndarray, next_frame: np.ndarray) -> np.ndarray:
    """Compute pixel-level absolute difference magnitude between two BGR frames, returning (H, W) float32."""
    assert prev_frame.shape == next_frame.shape, f"Frame shape mismatch: {prev_frame.shape} vs {next_frame.shape}"
    assert prev_frame.ndim == 3 and prev_frame.shape[2] == 3, f"Expected (H, W, 3), got {prev_frame.shape}"
    assert prev_frame.dtype == np.uint8 and next_frame.dtype == np.uint8, "Frames must be uint8"

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(next_gray - prev_gray)
    return diff


# Module-level cache for learned models to avoid reload overhead
_LEARNED_FLOW_MODEL = None
_FASTFLOWNET_MODEL = None


def flow_fastflownet(prev_frame: np.ndarray, next_frame: np.ndarray, weights_path: str = "weights/fastflownet_ft_mix.pth") -> np.ndarray:
    """Compute dense optical flow between two BGR frames using FastFlowNet on GPU/CPU, returning (H, W, 2) float32."""
    global _FASTFLOWNET_MODEL
    assert prev_frame.shape == next_frame.shape, f"Frame shape mismatch: {prev_frame.shape} vs {next_frame.shape}"
    h, w = prev_frame.shape[:2]

    if _FASTFLOWNET_MODEL is None:
        from src.fastflownet import load_fastflownet
        _FASTFLOWNET_MODEL = load_fastflownet(weights_path=weights_path)

    device = next(_FASTFLOWNET_MODEL.parameters()).device
    div_size = 64
    div_flow = 20.0
    input_h = int(div_size * np.ceil(h / div_size))
    input_w = int(div_size * np.ceil(w / div_size))

    t1 = torch.from_numpy(prev_frame[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    t2 = torch.from_numpy(next_frame[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
    rgb_mean = torch.cat([t1, t2], dim=2).view(1, 3, -1).mean(2).view(1, 3, 1, 1)
    t1, t2 = t1 - rgb_mean, t2 - rgb_mean

    if input_h != h or input_w != w:
        t1 = torch.nn.functional.interpolate(t1, size=(input_h, input_w), mode="bilinear", align_corners=False)
        t2 = torch.nn.functional.interpolate(t2, size=(input_h, input_w), mode="bilinear", align_corners=False)

    inp = torch.cat([t1, t2], dim=1)
    with torch.no_grad():
        out = _FASTFLOWNET_MODEL(inp)

    flow = div_flow * torch.nn.functional.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
    flow[:, 0, :, :] *= (w / float(input_w))
    flow[:, 1, :, :] *= (h / float(input_h))
    flow_np = flow[0].permute(1, 2, 0).cpu().numpy()
    return flow_np.astype(np.float32)


def flow_learned(prev_frame: np.ndarray, next_frame: np.ndarray, model: str = "fastflownet") -> np.ndarray:
    """Estimate optical flow using a learned model ('fastflownet' or 'raft'), returning (H, W, 2) float32."""
    if model.lower() in ["fastflownet", "fastflow"]:
        return flow_fastflownet(prev_frame, next_frame)

    global _LEARNED_FLOW_MODEL
    assert prev_frame.shape == next_frame.shape, f"Frame shape mismatch: {prev_frame.shape} vs {next_frame.shape}"
    h, w = prev_frame.shape[:2]

    if _LEARNED_FLOW_MODEL is None:
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights = Raft_Small_Weights.DEFAULT
        _LEARNED_FLOW_MODEL = raft_small(weights=weights).to(device).eval()

    device = next(_LEARNED_FLOW_MODEL.parameters()).device
    # Pad dimensions to multiples of 8 as required by RAFT feature encoders
    pad_h = (8 - (h % 8)) % 8
    pad_w = (8 - (w % 8)) % 8
    net_h, net_w = h + pad_h, w + pad_w

    img1 = cv2.resize(prev_frame, (net_w, net_h))
    img2 = cv2.resize(next_frame, (net_w, net_h))

    # Convert BGR (H, W, C) -> RGB (1, C, H, W) normalized to [-1, 1]
    t1 = torch.from_numpy(img1[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float().to(device)
    t2 = torch.from_numpy(img2[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float().to(device)
    t1 = (t1 / 127.5) - 1.0
    t2 = (t2 / 127.5) - 1.0

    with torch.no_grad():
        preds = _LEARNED_FLOW_MODEL(t1, t2)
        flow_pred = preds[-1].squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Resize flow back to original (H, W) and scale vectors by the aspect ratios (w/net_w, h/net_h)
    flow_resized = cv2.resize(flow_pred, (w, h), interpolation=cv2.INTER_LINEAR)
    flow_resized[:, :, 0] *= (w / float(net_w))
    flow_resized[:, :, 1] *= (h / float(net_h))

    return flow_resized.astype(np.float32)



def flow_to_speed_dir(flow: np.ndarray) -> tuple:
    """Compute magnitude speed (H, W) float32 and direction angle (H, W) float32 in [-pi, pi] from (H, W, 2) flow."""
    assert flow.ndim == 3 and flow.shape[2] == 2, f"Expected (H, W, 2) flow, got {flow.shape}"
    dx = flow[:, :, 0]
    dy = flow[:, :, 1]
    speed = np.sqrt(dx ** 2 + dy ** 2).astype(np.float32)
    angle = np.arctan2(dy, dx).astype(np.float32)
    return speed, angle
