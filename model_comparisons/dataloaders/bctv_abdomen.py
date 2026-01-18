"""
PyTorch DataLoader for BTCV abdominal organ dataset.
"""

import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
from pathlib import Path

from .utils import nifti_to_point_cloud


class BCTVAbdomenDataset(Dataset):
    """
    DataLoader for BTCV abdominal organ dataset.

    Structure:
        data/BCTV_abdomen/
            imagesTr/img{XXXX}.nii - Training CT scans
            labelsTr/label{XXXX}.nii - Training masks
            imagesTs/img{XXXX}.nii - Test CT scans (no labels)

    Returns:
        One triplet per (scan, organ_class) combination using midpoint slice of selected plane:
        2D slice (1, H, W), 2D binary mask (1, H, W), 3D point cloud (N, 3)
    """

    # Organ label mapping
    ORGAN_LABELS = {
        1: "Spleen",
        2: "Right Kidney",
        3: "Left Kidney",
        4: "Gallbladder",
        5: "Esophagus",
        6: "Liver",
        7: "Stomach",
        8: "Aorta",
        9: "IVC",
        10: "Veins",
        11: "Pancreas",
        12: "Right Adrenal",
        13: "Left Adrenal",
    }

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
        self.root_dir = Path(root_dir) / "BCTV_abdomen"
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
            self.label_dir = None  # No labels for test set

        # Collect all samples (one sample per scan per organ class)
        self.samples = []  # List of (image_path, num_slices, organ_label)

        image_files = sorted(self.image_dir.glob("img*.nii"))

        for image_path in image_files:
            # Extract image ID (e.g., "0001" from "img0001.nii")
            img_id = image_path.stem.replace("img", "")

            # Check if label exists for training
            if split == "train":
                label_path = self.label_dir / f"label{img_id}.nii"
                if not label_path.exists():
                    continue

                # Load mask to find which organs are present in this scan
                mask_nii = nib.load(str(label_path))
                mask_data = mask_nii.get_fdata()
                unique_labels = np.unique(mask_data).astype(int)

                # Load header to get number of slices
                img = nib.load(str(image_path))
                axis = self.plane_axis[self.plane]
                num_slices = img.shape[axis]
                slice_idx = num_slices // 2

                # Create one sample per organ present in this scan
                for organ_label in unique_labels:
                    if organ_label > 0 and organ_label in self.ORGAN_LABELS:
                        # Extract 2D mask at midpoint slice to check if organ is visible
                        organ_mask = (mask_data == organ_label)
                        if axis == 0:  # sagittal
                            mask_2d = organ_mask[slice_idx, :, :]
                        elif axis == 1:  # coronal
                            mask_2d = organ_mask[:, slice_idx, :]
                        else:  # axial
                            mask_2d = organ_mask[:, :, slice_idx]

                        # Skip if the 2D mask is all zeros (organ not visible in this slice)
                        if not np.any(mask_2d):
                            continue

                        self.samples.append((image_path, num_slices, organ_label))
            else:
                # Test set: no labels, just store scan
                img = nib.load(str(image_path))
                axis = self.plane_axis[self.plane]
                num_slices = img.shape[axis]
                self.samples.append((image_path, num_slices, None))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        image_path, num_slices, organ_label = self.samples[idx]

        # Calculate midpoint slice index
        slice_idx = num_slices // 2

        # Extract image ID
        img_id = image_path.stem.replace("img", "")

        # Load scan
        scan_nii = nib.load(str(image_path))
        scan_data = scan_nii.get_fdata()
        spacing = scan_nii.header.get_zooms()[:3]

        # Load mask if available
        if self.split == "train" and organ_label is not None:
            label_path = self.label_dir / f"label{img_id}.nii"
            mask_nii = nib.load(str(label_path))
            mask_data = mask_nii.get_fdata()

            # Filter by the specific organ label for this sample
            mask_data = (mask_data == organ_label).astype(np.float32)
        else:
            # No mask for test set - return zeros
            mask_data = np.zeros_like(scan_data)
            organ_label = 0

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

        return slice_2d, mask_2d, point_cloud, organ_label
