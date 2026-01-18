"""
PyTorch DataLoader for AeroPath respiratory tract dataset.
"""

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
from pathlib import Path

from .utils import nifti_to_point_cloud


class AeroPathDataset(Dataset):
    """
    DataLoader for AeroPath respiratory tract dataset.

    Structure:
        data/AeroPath/AeroPath/AeroPath/{patient_id}/
            {patient_id}_CT_HR.nii.gz - CT scan
            {patient_id}_CT_HR_label_airways.nii.gz - Airways mask
            {patient_id}_CT_HR_label_lungs.nii.gz - Lungs mask

    Returns:
        One triplet per (scan, mask_type) combination using midpoint slice of selected plane:
        2D slice (1, H, W), 2D binary mask (1, H, W), 3D surface point cloud (N, 3)
    """

    # Mask type mapping
    MASK_TYPES = {
        "airways": "Airways",
        "lungs": "Lungs",
    }

    def __init__(
        self,
        root_dir: str,
        plane: str = "coronal",
        transform: Optional[callable] = None,
        surface_only: bool = True,
    ):
        """
        Args:
            root_dir: Path to data directory (e.g., "data/")
            plane: Slice plane - "coronal" (frontal), "sagittal", or "axial" (transverse)
            transform: Optional transform to apply to 2D slice
            surface_only: If True, extract only surface points (exterior boundary).
                         If False, return all voxels.
        """
        self.root_dir = Path(root_dir) / "AeroPath" / "AeroPath" / "AeroPath"
        self.plane = plane.lower()
        self.transform = transform
        self.surface_only = surface_only

        # Map plane to axis index
        self.plane_axis = {"coronal": 1, "sagittal": 0, "axial": 2}
        if self.plane not in self.plane_axis:
            raise ValueError(f"plane must be one of {list(self.plane_axis.keys())}")

        # Collect all samples (one sample per scan per mask type)
        self.samples = []  # List of (patient_dir, num_slices, mask_type)

        patient_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])

        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            scan_path = patient_dir / f"{patient_id}_CT_HR.nii.gz"

            if scan_path.exists():
                # Load header to get number of slices without loading full volume
                img = nib.load(str(scan_path))
                axis = self.plane_axis[self.plane]
                num_slices = img.shape[axis]

                # Check which mask types exist and create a sample for each
                for mask_type in self.MASK_TYPES.keys():
                    mask_path = patient_dir / f"{patient_id}_CT_HR_label_{mask_type}.nii.gz"
                    if mask_path.exists():
                        self.samples.append((patient_dir, num_slices, mask_type))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        patient_dir, num_slices, mask_type = self.samples[idx]
        patient_id = patient_dir.name

        # Calculate midpoint slice index for selected plane
        slice_idx = num_slices // 2

        # Load NIfTI files
        scan_path = patient_dir / f"{patient_id}_CT_HR.nii.gz"
        mask_path = patient_dir / f"{patient_id}_CT_HR_label_{mask_type}.nii.gz"

        scan_nii = nib.load(str(scan_path))
        mask_nii = nib.load(str(mask_path))

        scan_data = scan_nii.get_fdata()
        mask_data = mask_nii.get_fdata()

        # Get voxel spacing for point cloud
        spacing = scan_nii.header.get_zooms()[:3]

        # Extract 2D slice at midpoint based on plane
        axis = self.plane_axis[self.plane]
        if axis == 0:  # sagittal
            slice_2d = scan_data[slice_idx, :, :].astype(np.float32)
            mask_2d = (mask_data[slice_idx, :, :] > 0).astype(np.float32)
        elif axis == 1:  # coronal (frontal)
            slice_2d = scan_data[:, slice_idx, :].astype(np.float32)
            mask_2d = (mask_data[:, slice_idx, :] > 0).astype(np.float32)
        else:  # axial (transverse)
            slice_2d = scan_data[:, :, slice_idx].astype(np.float32)
            mask_2d = (mask_data[:, :, slice_idx] > 0).astype(np.float32)

        # Generate 3D point cloud from full mask volume (surface points only by default)
        point_cloud = nifti_to_point_cloud(mask_data, spacing, surface_only=self.surface_only)

        # Convert to tensors
        slice_2d = torch.from_numpy(slice_2d).unsqueeze(0)  # (1, H, W)
        mask_2d = torch.from_numpy(mask_2d).unsqueeze(0)    # (1, H, W)
        point_cloud = torch.from_numpy(point_cloud)          # (N, 3)

        if self.transform:
            slice_2d = self.transform(slice_2d)

        return slice_2d, mask_2d, point_cloud, mask_type
