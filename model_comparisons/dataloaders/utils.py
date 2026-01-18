"""
Utility functions for medical imaging dataloaders.
"""

import numpy as np
import torch
from scipy import ndimage
from typing import Tuple, List


def extract_surface_points(
    mask_volume: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> np.ndarray:
    """
    Extract only the surface (exterior) points from a 3D binary mask.
    Uses morphological erosion to find boundary voxels.

    Args:
        mask_volume: 3D numpy array with binary mask
        spacing: Voxel spacing (x, y, z) to scale coordinates

    Returns:
        Surface point cloud as numpy array of shape (N, 3)
    """
    # Ensure binary mask
    binary_mask = (mask_volume > 0).astype(np.uint8)

    if binary_mask.sum() == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Create structuring element for 3D erosion (6-connectivity)
    struct = ndimage.generate_binary_structure(3, 1)

    # Erode the mask
    eroded = ndimage.binary_erosion(binary_mask, structure=struct)

    # Surface = original mask minus eroded mask (boundary voxels)
    surface_mask = binary_mask.astype(bool) & ~eroded

    # Get coordinates of surface voxels
    coords = np.argwhere(surface_mask)

    if len(coords) == 0:
        # If erosion removed everything (very thin structure), use original
        coords = np.argwhere(binary_mask > 0)

    # Scale by voxel spacing
    point_cloud = coords.astype(np.float32) * np.array(spacing, dtype=np.float32)

    return point_cloud


def nifti_to_point_cloud(
    mask_volume: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    surface_only: bool = True
) -> np.ndarray:
    """
    Convert a 3D binary mask to a point cloud.

    Args:
        mask_volume: 3D numpy array with binary mask
        spacing: Voxel spacing (x, y, z) to scale coordinates
        surface_only: If True, extract only surface/exterior points.
                     If False, return all non-zero voxels.

    Returns:
        Point cloud as numpy array of shape (N, 3)
    """
    if surface_only:
        return extract_surface_points(mask_volume, spacing)

    # Get coordinates of all non-zero voxels
    coords = np.argwhere(mask_volume > 0)

    if len(coords) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    # Scale by voxel spacing
    point_cloud = coords.astype(np.float32) * np.array(spacing, dtype=np.float32)

    return point_cloud


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, any]]):
    """
    Custom collate function to handle variable-size point clouds.

    Args:
        batch: List of (slice_2d, mask_2d, point_cloud, class_label) tuples

    Returns:
        Batched slices, batched masks, list of point clouds, list of class labels
    """
    slices, masks, point_clouds, class_labels = zip(*batch)
    slices = torch.stack(slices, dim=0)
    masks = torch.stack(masks, dim=0)
    # Point clouds have variable sizes, keep as list
    return slices, masks, list(point_clouds), list(class_labels)
