"""
PyTorch DataLoader for MSD dataset.
"""

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
from pathlib import Path

from .utils import nifti_to_point_cloud


class MSDDataset(Dataset):
    """
    DataLoader for MSD dataset.

    Structure:
        data/MSD_lung/
            imagesTr/lung_{XXX}.nii/lung_{XXX}.nii - Training CT scans
            labelsTr/lung_{XXX}.nii/lung_{XXX}.nii - Training masks
            imagesTs/lung_{XXX}.nii/lung_{XXX}.nii - Test CT scans

    Returns:
        One triplet per (scan, class) combination using midpoint slice of selected plane:
        2D slice (1, H, W), 2D binary mask (1, H, W), 3D point cloud (N, 3)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        plane: str = "coronal",
        transform: Optional[callable] = None,
    ):
        """
        Args:
            root_dir: Path to data directory (e.g., "data/")
            split: "train" or "test"
            plane: Slice plane - "coronal" (frontal), "sagittal", or "axial" (transverse)
            transform: Optional transform to apply to 2D slice
        """
        self.root_dir = Path(root_dir) / "MSD_lung"
        self.split = split
        self.plane = plane.lower()
        self.transform = transform

        # Map plane to axis index
        self.plane_axis = {"coronal": 1, "sagittal": 0, "axial": 2}
        if self.plane not in self.plane_axis:
            raise ValueError(f"plane must be one of {list(self.plane_axis.keys())}")

        # Set paths based on split
        if split == "train":
            self.image_dir = self.root_dir / "imagesTr"
            self.label_dir = self.root_dir / "labelsTr"
        else:
            self.image_dir = self.root_dir / "imagesTs"
            self.label_dir = None

        # Collect all samples (one sample per scan per class)
        self.samples = []  # List of (image_path, num_slices, class_label)

        # Handle the nested directory structure (lung_XXX.nii/lung_XXX.nii)
        for entry in sorted(self.image_dir.iterdir()):
            if entry.is_dir():
                # Look for .nii or .nii.gz file inside
                nii_files = list(entry.glob("*.nii")) + list(entry.glob("*.nii.gz"))
                if nii_files:
                    image_path = nii_files[0]

                    # Check label exists for training
                    if split == "train":
                        label_entry = self.label_dir / entry.name
                        if label_entry.exists():
                            label_files = list(label_entry.glob("*.nii")) + list(label_entry.glob("*.nii.gz"))
                            if not label_files:
                                continue

                            # Load mask to find unique class labels
                            mask_nii = nib.load(str(label_files[0]))
                            mask_data = mask_nii.get_fdata()
                            unique_labels = np.unique(mask_data).astype(int)

                            # Load header to get number of slices
                            img = nib.load(str(image_path))
                            axis = self.plane_axis[self.plane]
                            num_slices = img.shape[axis]
                            slice_idx = num_slices // 2

                            # Create one sample per class present in this scan
                            for class_label in unique_labels:
                                if class_label > 0:
                                    # Extract 2D mask at midpoint slice to check if class is visible
                                    class_mask = (mask_data == class_label)
                                    if axis == 0:  # sagittal
                                        mask_2d = class_mask[slice_idx, :, :]
                                    elif axis == 1:  # coronal
                                        mask_2d = class_mask[:, slice_idx, :]
                                    else:  # axial
                                        mask_2d = class_mask[:, :, slice_idx]

                                    # Skip if the 2D mask is all zeros (class not visible in this slice)
                                    if not np.any(mask_2d):
                                        continue

                                    self.samples.append((image_path, num_slices, class_label))
                    else:
                        # Test set: no labels
                        img = nib.load(str(image_path))
                        axis = self.plane_axis[self.plane]
                        num_slices = img.shape[axis]
                        self.samples.append((image_path, num_slices, None))

    def __len__(self) -> int:
        return len(self.samples)

    def _find_label_path(self, image_path: Path) -> Optional[Path]:
        """Find corresponding label file for an image."""
        # Get the directory name (e.g., "lung_001.nii")
        dir_name = image_path.parent.name
        label_dir = self.label_dir / dir_name

        if label_dir.exists():
            label_files = list(label_dir.glob("*.nii")) + list(label_dir.glob("*.nii.gz"))
            if label_files:
                return label_files[0]
        return None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        image_path, num_slices, class_label = self.samples[idx]

        # Calculate midpoint slice index
        slice_idx = num_slices // 2

        # Load scan
        scan_nii = nib.load(str(image_path))
        scan_data = scan_nii.get_fdata()
        spacing = scan_nii.header.get_zooms()[:3]

        # Load mask if available
        if self.split == "train" and class_label is not None:
            label_path = self._find_label_path(image_path)
            if label_path:
                mask_nii = nib.load(str(label_path))
                mask_data = (mask_nii.get_fdata() == class_label).astype(np.float32)
            else:
                mask_data = np.zeros_like(scan_data)
                class_label = 0
        else:
            mask_data = np.zeros_like(scan_data)
            class_label = 0

        # Extract 2D slice at midpoint based on plane
        axis = self.plane_axis[self.plane]
        if axis == 0:  # sagittal
            slice_2d = scan_data[slice_idx, :, :].astype(np.float32)
            mask_2d = mask_data[slice_idx, :, :].astype(np.float32)
        elif axis == 1:  # coronal (frontal)
            slice_2d = scan_data[:, slice_idx, :].astype(np.float32)
            mask_2d = mask_data[:, slice_idx, :].astype(np.float32)
        else:  # axial (transverse)
            slice_2d = scan_data[:, :, slice_idx].astype(np.float32)
            mask_2d = mask_data[:, :, slice_idx].astype(np.float32)

        # Generate 3D point cloud
        point_cloud = nifti_to_point_cloud(mask_data, spacing)

        # Convert to tensors
        slice_2d = torch.from_numpy(slice_2d).unsqueeze(0)  # (1, H, W)
        mask_2d = torch.from_numpy(mask_2d).unsqueeze(0)    # (1, H, W)
        point_cloud = torch.from_numpy(point_cloud)          # (N, 3)

        if self.transform:
            slice_2d = self.transform(slice_2d)

        return slice_2d, mask_2d, point_cloud, class_label
