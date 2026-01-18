"""
3D Reconstruction Evaluation Metrics.

Metrics:
1. F-score@0.01 - Harmonic mean of precision/recall at 0.01 distance threshold
2. Voxel-IoU - Intersection over Union on 64^3 voxel grid
3. Chamfer Distance (CD) - Bidirectional nearest-neighbor distances
4. Earth Mover's Distance (EMD) - Minimum-cost transport between point sets
5. Precision - Fraction of predicted points within threshold of GT
6. Recall - Fraction of GT points within threshold of prediction
7. Voxel-Dice - Dice coefficient on voxel grid (sensitive to small structures)
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from typing import Dict, Tuple, Optional


def icp_align(
    source: np.ndarray,
    target: np.ndarray,
    max_iterations: int = 50,
    tolerance: float = 1e-6,
    n_samples: int = 5000
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Align source point cloud to target using Iterative Closest Point (ICP).

    Args:
        source: Source point cloud to be aligned (N, 3)
        target: Target point cloud (M, 3)
        max_iterations: Maximum ICP iterations
        tolerance: Convergence tolerance for transformation change
        n_samples: Number of points to subsample for efficiency

    Returns:
        aligned_source: Aligned source point cloud
        transformation: 4x4 transformation matrix
        final_error: Final mean squared error
    """
    # Subsample for efficiency
    if len(source) > n_samples:
        idx = np.random.choice(len(source), n_samples, replace=False)
        source_sample = source[idx]
    else:
        source_sample = source.copy()

    if len(target) > n_samples:
        idx = np.random.choice(len(target), n_samples, replace=False)
        target_sample = target[idx]
    else:
        target_sample = target.copy()

    # Initialize transformation
    current_source = source_sample.copy()
    transformation = np.eye(4)
    prev_error = float('inf')

    for iteration in range(max_iterations):
        # Find nearest neighbors
        tree = cKDTree(target_sample)
        distances, indices = tree.query(current_source, k=1)
        matched_target = target_sample[indices]

        # Compute centroids
        centroid_source = current_source.mean(axis=0)
        centroid_target = matched_target.mean(axis=0)

        # Center the point clouds
        source_centered = current_source - centroid_source
        target_centered = matched_target - centroid_target

        # Compute rotation using SVD
        H = source_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # Handle reflection case
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # Compute translation
        t = centroid_target - R @ centroid_source

        # Apply transformation
        current_source = (R @ current_source.T).T + t

        # Update cumulative transformation
        T_step = np.eye(4)
        T_step[:3, :3] = R
        T_step[:3, 3] = t
        transformation = T_step @ transformation

        # Check convergence
        mean_error = distances.mean()
        if abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # Apply final transformation to full source point cloud
    aligned_source = (transformation[:3, :3] @ source.T).T + transformation[:3, 3]

    return aligned_source, transformation, prev_error


def normalize_point_clouds(
    pc_pred: np.ndarray,
    pc_gt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize both point clouds to unit cube centered at origin.
    Uses GT bounds for consistent normalization.
    """
    # Use GT bounds for normalization
    center = (pc_gt.max(axis=0) + pc_gt.min(axis=0)) / 2
    scale = (pc_gt.max(axis=0) - pc_gt.min(axis=0)).max()

    if scale == 0:
        scale = 1.0

    pc_pred_norm = (pc_pred - center) / scale
    pc_gt_norm = (pc_gt - center) / scale

    return pc_pred_norm, pc_gt_norm


def compute_precision_recall(
    pc_pred: np.ndarray,
    pc_gt: np.ndarray,
    threshold: float = 0.01
) -> Tuple[float, float]:
    """
    Compute point-level precision and recall.

    Precision: fraction of predicted points within threshold of any GT point
    Recall: fraction of GT points within threshold of any predicted point

    Args:
        pc_pred: Predicted point cloud (N, 3)
        pc_gt: Ground truth point cloud (M, 3)
        threshold: Distance threshold

    Returns:
        (precision, recall)
    """
    if len(pc_pred) == 0 or len(pc_gt) == 0:
        return 0.0, 0.0

    # Build KD-trees
    tree_gt = cKDTree(pc_gt)
    tree_pred = cKDTree(pc_pred)

    # Precision: for each predicted point, find distance to nearest GT point
    dist_pred_to_gt, _ = tree_gt.query(pc_pred, k=1)
    precision = (dist_pred_to_gt <= threshold).mean()

    # Recall: for each GT point, find distance to nearest predicted point
    dist_gt_to_pred, _ = tree_pred.query(pc_gt, k=1)
    recall = (dist_gt_to_pred <= threshold).mean()

    return float(precision), float(recall)


def compute_fscore(
    pc_pred: np.ndarray,
    pc_gt: np.ndarray,
    threshold: float = 0.01
) -> float:
    """
    Compute F-score at given threshold.

    F-score = 2 * (precision * recall) / (precision + recall)

    Args:
        pc_pred: Predicted point cloud (N, 3)
        pc_gt: Ground truth point cloud (M, 3)
        threshold: Distance threshold (default 0.01)

    Returns:
        F-score value
    """
    precision, recall = compute_precision_recall(pc_pred, pc_gt, threshold)

    if precision + recall == 0:
        return 0.0

    fscore = 2 * (precision * recall) / (precision + recall)
    return float(fscore)


def compute_chamfer_distance(
    pc_pred: np.ndarray,
    pc_gt: np.ndarray
) -> float:
    """
    Compute Chamfer Distance between two point clouds.

    CD = mean(min_dist(pred->gt)) + mean(min_dist(gt->pred))

    Args:
        pc_pred: Predicted point cloud (N, 3)
        pc_gt: Ground truth point cloud (M, 3)

    Returns:
        Chamfer distance value
    """
    if len(pc_pred) == 0 or len(pc_gt) == 0:
        return float('inf')

    tree_gt = cKDTree(pc_gt)
    tree_pred = cKDTree(pc_pred)

    # Pred to GT distances
    dist_pred_to_gt, _ = tree_gt.query(pc_pred, k=1)

    # GT to pred distances
    dist_gt_to_pred, _ = tree_pred.query(pc_gt, k=1)

    # Chamfer distance (using squared distances is common, but we use L2)
    cd = dist_pred_to_gt.mean() + dist_gt_to_pred.mean()

    return float(cd)


def compute_emd(
    pc_pred: np.ndarray,
    pc_gt: np.ndarray,
    n_samples: int = 1024
) -> float:
    """
    Compute Earth Mover's Distance (approximate) between two point clouds.

    Uses Hungarian algorithm on subsampled point clouds for efficiency.

    Args:
        pc_pred: Predicted point cloud (N, 3)
        pc_gt: Ground truth point cloud (M, 3)
        n_samples: Number of points to subsample for efficiency

    Returns:
        EMD value
    """
    if len(pc_pred) == 0 or len(pc_gt) == 0:
        return float('inf')

    # Subsample for efficiency
    if len(pc_pred) > n_samples:
        idx = np.random.choice(len(pc_pred), n_samples, replace=False)
        pc_pred = pc_pred[idx]

    if len(pc_gt) > n_samples:
        idx = np.random.choice(len(pc_gt), n_samples, replace=False)
        pc_gt = pc_gt[idx]

    # Make both point clouds same size
    n = min(len(pc_pred), len(pc_gt))
    pc_pred = pc_pred[:n]
    pc_gt = pc_gt[:n]

    # Compute pairwise distances
    # Using scipy's cdist would be cleaner but let's keep dependencies minimal
    diff = pc_pred[:, np.newaxis, :] - pc_gt[np.newaxis, :, :]
    cost_matrix = np.sqrt((diff ** 2).sum(axis=2))

    # Hungarian algorithm for optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # EMD is mean of matched distances
    emd = cost_matrix[row_ind, col_ind].mean()

    return float(emd)


def voxelize_point_cloud(
    pc: np.ndarray,
    grid_size: int = 64,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> np.ndarray:
    """
    Convert point cloud to voxel grid.

    Args:
        pc: Point cloud (N, 3)
        grid_size: Size of voxel grid
        bounds: Optional (min_bound, max_bound) for consistent voxelization

    Returns:
        Binary voxel grid (grid_size, grid_size, grid_size)
    """
    if len(pc) == 0:
        return np.zeros((grid_size, grid_size, grid_size), dtype=bool)

    if bounds is None:
        min_bound = pc.min(axis=0)
        max_bound = pc.max(axis=0)
    else:
        min_bound, max_bound = bounds

    # Add small epsilon to avoid edge cases
    extent = max_bound - min_bound
    extent = np.maximum(extent, 1e-6)

    # Normalize to [0, grid_size-1]
    normalized = (pc - min_bound) / extent * (grid_size - 1)
    normalized = np.clip(normalized, 0, grid_size - 1).astype(int)

    # Create voxel grid
    voxels = np.zeros((grid_size, grid_size, grid_size), dtype=bool)
    voxels[normalized[:, 0], normalized[:, 1], normalized[:, 2]] = True

    return voxels


def compute_voxel_iou(
    pc_pred: np.ndarray,
    pc_gt: np.ndarray,
    grid_size: int = 64
) -> float:
    """
    Compute Voxel IoU between two point clouds.

    Args:
        pc_pred: Predicted point cloud (N, 3)
        pc_gt: Ground truth point cloud (M, 3)
        grid_size: Size of voxel grid

    Returns:
        IoU value
    """
    if len(pc_pred) == 0 or len(pc_gt) == 0:
        return 0.0

    # Use combined bounds for consistent voxelization
    all_points = np.vstack([pc_pred, pc_gt])
    min_bound = all_points.min(axis=0)
    max_bound = all_points.max(axis=0)
    bounds = (min_bound, max_bound)

    voxels_pred = voxelize_point_cloud(pc_pred, grid_size, bounds)
    voxels_gt = voxelize_point_cloud(pc_gt, grid_size, bounds)

    intersection = (voxels_pred & voxels_gt).sum()
    union = (voxels_pred | voxels_gt).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def compute_voxel_dice(
    pc_pred: np.ndarray,
    pc_gt: np.ndarray,
    grid_size: int = 64
) -> float:
    """
    Compute Voxel Dice coefficient between two point clouds.

    Dice = 2 * |A ∩ B| / (|A| + |B|)

    More sensitive to small structures than IoU.

    Args:
        pc_pred: Predicted point cloud (N, 3)
        pc_gt: Ground truth point cloud (M, 3)
        grid_size: Size of voxel grid

    Returns:
        Dice coefficient
    """
    if len(pc_pred) == 0 or len(pc_gt) == 0:
        return 0.0

    # Use combined bounds for consistent voxelization
    all_points = np.vstack([pc_pred, pc_gt])
    min_bound = all_points.min(axis=0)
    max_bound = all_points.max(axis=0)
    bounds = (min_bound, max_bound)

    voxels_pred = voxelize_point_cloud(pc_pred, grid_size, bounds)
    voxels_gt = voxelize_point_cloud(pc_gt, grid_size, bounds)

    intersection = (voxels_pred & voxels_gt).sum()
    sum_voxels = voxels_pred.sum() + voxels_gt.sum()

    if sum_voxels == 0:
        return 0.0

    return float(2 * intersection / sum_voxels)


def evaluate_reconstruction(
    pc_pred: np.ndarray,
    pc_gt: np.ndarray,
    threshold: float = 0.01,
    grid_size: int = 64,
    normalize: bool = True,
    align: bool = True
) -> Dict[str, float]:
    """
    Compute all 7 reconstruction evaluation metrics.

    Args:
        pc_pred: Predicted point cloud (N, 3)
        pc_gt: Ground truth point cloud (M, 3)
        threshold: Distance threshold for F-score and precision/recall
        grid_size: Voxel grid size for IoU and Dice
        normalize: Whether to normalize point clouds to unit cube
        align: Whether to auto-align predicted to GT using ICP

    Returns:
        Dictionary with all metrics:
        - fscore: F-score@threshold
        - voxel_iou: Voxel IoU
        - chamfer_distance: Chamfer Distance
        - emd: Earth Mover's Distance
        - precision: Point-level precision
        - recall: Point-level recall
        - voxel_dice: Voxel Dice coefficient
    """
    pc_pred = np.asarray(pc_pred)
    pc_gt = np.asarray(pc_gt)

    if normalize:
        pc_pred, pc_gt = normalize_point_clouds(pc_pred, pc_gt)

    if align:
        pc_pred, _, _ = icp_align(pc_pred, pc_gt)

    precision, recall = compute_precision_recall(pc_pred, pc_gt, threshold)

    metrics = {
        'fscore': compute_fscore(pc_pred, pc_gt, threshold),
        'voxel_iou': compute_voxel_iou(pc_pred, pc_gt, grid_size),
        'chamfer_distance': compute_chamfer_distance(pc_pred, pc_gt),
        'emd': compute_emd(pc_pred, pc_gt),
        'precision': precision,
        'recall': recall,
        'voxel_dice': compute_voxel_dice(pc_pred, pc_gt, grid_size),
    }

    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Reconstruction Metrics"):
    """Pretty print metrics."""
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")
    print(f"{'Metric':<25} {'Value':>20}")
    print(f"{'-'*50}")
    print(f"{'F-score@0.01':<25} {metrics['fscore']:>20.4f}")
    print(f"{'Voxel-IoU':<25} {metrics['voxel_iou']:>20.4f}")
    print(f"{'Chamfer Distance':<25} {metrics['chamfer_distance']:>20.4f}")
    print(f"{'Earth Mover Distance':<25} {metrics['emd']:>20.4f}")
    print(f"{'Precision':<25} {metrics['precision']:>20.4f}")
    print(f"{'Recall':<25} {metrics['recall']:>20.4f}")
    print(f"{'Voxel-Dice':<25} {metrics['voxel_dice']:>20.4f}")
    print(f"{'='*50}\n")
