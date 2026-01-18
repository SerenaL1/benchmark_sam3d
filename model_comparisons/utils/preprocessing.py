"""
Image preprocessing utilities for CT slices.

Functions for preparing medical images for 3D reconstruction models.
"""

import numpy as np
from PIL import Image
import torch


def prepare_slice_for_hunyuan(
    slice_tensor: torch.Tensor,
    mask_tensor: torch.Tensor = None,
    use_mask_overlay: bool = True,
    window_center: int = -600,
    window_width: int = 1500,
    rotate: bool = True
) -> Image.Image:
    """
    Convert a CT slice tensor to a PIL Image suitable for Hunyuan3D.

    Args:
        slice_tensor: (1, H, W) tensor with CT values
        mask_tensor: Optional (1, H, W) binary mask tensor
        use_mask_overlay: If True and mask provided, apply mask to extract region of interest
        window_center: CT window center (default -600 for lung)
        window_width: CT window width (default 1500 for lung)
        rotate: If True, rotate 90 degrees CCW for correct orientation

    Returns:
        PIL Image in RGBA format
    """
    # Get numpy array
    slice_np = slice_tensor.squeeze().numpy()

    # Rotate 90 degrees CCW for correct orientation if requested
    if rotate:
        slice_np = np.rot90(slice_np)

    # Apply CT windowing
    vmin = window_center - window_width // 2
    vmax = window_center + window_width // 2

    slice_np = np.clip(slice_np, vmin, vmax)
    slice_np = ((slice_np - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    # Create RGB image
    rgb = np.stack([slice_np, slice_np, slice_np], axis=-1)

    # Apply mask - keep only the masked region with transparency elsewhere
    if mask_tensor is not None and use_mask_overlay:
        mask_np = mask_tensor.squeeze().numpy()
        if rotate:
            mask_np = np.rot90(mask_np)
        mask_np = mask_np.astype(np.uint8)
        # Alpha channel: 255 where mask is present, 0 elsewhere
        alpha = (mask_np * 255).astype(np.uint8)
    else:
        # Full opacity if no mask
        alpha = np.ones_like(slice_np, dtype=np.uint8) * 255

    rgba = np.dstack([rgb, alpha]).astype(np.uint8)

    return Image.fromarray(rgba, mode='RGBA')


def normalize_to_unit_range(points: np.ndarray) -> np.ndarray:
    """
    Normalize a point cloud to [-1, 1] range using its own bounds.

    Args:
        points: Point cloud (N, 3)

    Returns:
        Normalized point cloud with coordinates in [-1, 1]
    """
    center = (points.max(axis=0) + points.min(axis=0)) / 2
    scale = (points.max(axis=0) - points.min(axis=0)).max() / 2

    if scale == 0:
        scale = 1.0

    return (points - center) / scale
