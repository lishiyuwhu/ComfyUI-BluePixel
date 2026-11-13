"""Image preprocessing utilities bridging ComfyUI IMAGE to BluePixel inputs.

ComfyUI IMAGE: torch tensor (B, H, W, C), float32, [0,1], RGB.
BluePixel expects PIL or (1, 3, H, W) in [-1, 1].
"""
from __future__ import annotations

import torch

MAX_BATCH = 4096


def comfy_image_to_tensor(image: torch.Tensor) -> torch.Tensor:
    """ComfyUI (B,H,W,C) [0,1] -> (B,3,H,W) [-1,1] float32."""
    if image.ndim != 4:
        raise ValueError(f"expected NHWC image tensor, got shape {tuple(image.shape)}")
    if image.shape[-1] != 3:
        raise ValueError(f"expected 3 channels last, got {image.shape[-1]}")
    x = image.permute(0, 3, 1, 2).to(torch.float32)
    if x.min() < 0.0 - 1e-6 or x.max() > 1.0 + 1e-6:
        raise ValueError(
            f"IMAGE values must be in [0,1], got [{x.min().item():.4f}, {x.max().item():.4f}]"
        )
    return x * 2.0 - 1.0


def tensor_to_comfy_image(tensor: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) in [-1,1] -> (B,H,W,C) [0,1] float32 CPU."""
    x = tensor.detach().to(torch.float32).clamp(-1.0, 1.0)
    x = (x + 1.0) / 2.0
    return x.permute(0, 2, 3, 1).contiguous().cpu()


def tensor_to_pil(tensor: torch.Tensor):
    """(1,3,H,W) in [-1,1] -> PIL Image with round-to-nearest uint8 quantization.

    torchvision's to_pil_image truncates float->uint8 which loses up to 1 LSB;
    we round to match the original uint8 pixels exactly (needed for bit-exact
    parity with the reference PIL.LANCZOS resize path).
    """
    from PIL import Image
    import numpy as np

    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(f"expected (1,3,H,W), got {tuple(tensor.shape)}")
    x = ((tensor[0].clamp(-1, 1) + 1.0) / 2.0 * 255.0 + 0.5)
    arr = x.to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


def pil_to_comfy_image(pil_image) -> torch.Tensor:
    """PIL -> ComfyUI (1,H,W,3) [0,1]."""
    import numpy as np

    arr = np.asarray(pil_image.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)
