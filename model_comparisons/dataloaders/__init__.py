"""
PyTorch DataLoaders for medical imaging datasets.
Each loader returns: 2D slice, 2D binary mask, 3D point cloud.
"""

import torch
from torch.utils.data import DataLoader

from .aeropath import AeroPathDataset
from .bctv_abdomen import BCTVAbdomenDataset
from .msd import MSDDataset
from .utils import nifti_to_point_cloud, collate_fn


__all__ = [
    "AeroPathDataset",
    "BCTVAbdomenDataset",
    "MSDDataset",
    "nifti_to_point_cloud",
    "collate_fn",
    "get_dataloader",
]


def get_dataloader(
    dataset_name: str,
    root_dir: str = "data",
    batch_size: int = 1,
    num_workers: int = 0,
    **dataset_kwargs
) -> DataLoader:
    """
    Factory function to get a DataLoader for any of the three datasets.

    Args:
        dataset_name: One of "aeropath", "bctv", "msd_lung"
        root_dir: Path to data directory
        batch_size: Batch size (default 1 due to variable point cloud sizes)
        num_workers: Number of data loading workers
        **dataset_kwargs: Additional arguments for the specific dataset

    Returns:
        PyTorch DataLoader
    """
    dataset_map = {
        "aeropath": AeroPathDataset,
        "bctv": BCTVAbdomenDataset,
        "msd": MSDDataset,
    }

    if dataset_name.lower() not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(dataset_map.keys())}")

    dataset_class = dataset_map[dataset_name.lower()]
    dataset = dataset_class(root_dir=root_dir, **dataset_kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Inference mode
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
