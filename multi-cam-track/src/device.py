"""Hardware device resolution supporting Apple Silicon (MPS), CUDA, and CPU."""
import torch


def resolve_device(preferred='auto'):
    """Resolve compute device prioritizing Apple Silicon MPS, CUDA, or CPU.

    Args:
        preferred (str): Desired device ('auto', 'mps', 'cuda', 'cpu', etc.).
                         If 'auto', picks 'mps' on Apple Silicon, 'cuda' if available,
                         or falls back to 'cpu'.
                         If 'cuda' is requested but unavailable, falls back to 'mps' if present,
                         otherwise 'cpu'.

    Returns:
        str: Device identifier suitable for PyTorch and Ultralytics YOLO.
    """
    if preferred == 'auto':
        if torch.backends.mps.is_available():
            return 'mps'
        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    if preferred == 'cuda':
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    if preferred == 'mps':
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    return preferred or 'cpu'


def get_device_name(device_str):
    """Return a human-readable description of the resolved compute device."""
    if device_str == 'mps':
        return 'Apple Silicon GPU (MPS)'
    if device_str == 'cuda':
        device_idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
        name = torch.cuda.get_device_name(device_idx) if torch.cuda.is_available() else 'CUDA'
        return f'NVIDIA GPU ({name})'
    return 'CPU'

