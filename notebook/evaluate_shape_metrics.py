import numpy as np
import open3d as o3d
import nibabel as nib
from skimage import measure  # For Marching Cubes
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import os
from pathlib import Path
import pandas as pd
import argparse

# ==========================================
# 1. Metric Helper Functions (Preserved)
# ==========================================

def normalize_to_unit_cube(points):
    """Normalizes point cloud to [-1, 1] range."""
    if len(points) == 0: return points
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    max_dist = np.max(np.abs(points_centered))
    return points_centered / max_dist if max_dist > 0 else points_centered

def apply_icp(source_points, target_points, threshold=0.02):
    """Aligns source points to target points using ICP."""
    if len(source_points) == 0 or len(target_points) == 0:
        return source_points
    
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)
    
    trans_init = np.identity(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    source_pcd.transform(reg_p2p.transformation)
    return np.asarray(source_pcd.points)

def compute_f1_score(pred_points, gt_points, threshold=0.01):
    """Computes F1@0.01 score."""
    if len(pred_points) == 0 or len(gt_points) == 0: return 0.0, 0.0, 0.0
    
    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_points)
    
    dist_pred_to_gt, _ = gt_tree.query(pred_points, k=1)
    precision = np.mean(dist_pred_to_gt < threshold)
    
    dist_gt_to_pred, _ = pred_tree.query(gt_points, k=1)
    recall = np.mean(dist_gt_to_pred < threshold)
    
    if precision + recall == 0: return 0.0, precision, recall
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1, precision, recall

def compute_voxel_iou(point_cloud_1, point_cloud_2, resolution=64):
    """Computes Voxel-IoU at 64^3 resolution."""
    if len(point_cloud_1) == 0 and len(point_cloud_2) == 0: return 1.0
    if len(point_cloud_1) == 0 or len(point_cloud_2) == 0: return 0.0
    
    min_bound = np.array([-1.0, -1.0, -1.0])
    max_bound = np.array([1.0, 1.0, 1.0])
    voxel_size = (max_bound - min_bound) / resolution
    
    def get_voxels(points):
        indices = np.floor((points - min_bound) / voxel_size).astype(int)
        indices = np.clip(indices, 0, resolution - 1)
        return set(map(tuple, indices))

    voxels_1 = get_voxels(point_cloud_1)
    voxels_2 = get_voxels(point_cloud_2)
    
    intersection = len(voxels_1.intersection(voxels_2))
    union = len(voxels_1.union(voxels_2))
    return intersection / union if union > 0 else 0.0

def compute_voxel_dice(point_cloud_1, point_cloud_2, resolution=64):
    """Computes Voxel-Dice at 64^3 resolution."""
    if len(point_cloud_1) == 0 and len(point_cloud_2) == 0: return 1.0
    if len(point_cloud_1) == 0 or len(point_cloud_2) == 0: return 0.0
    
    min_bound = np.array([-1.0, -1.0, -1.0])
    max_bound = np.array([1.0, 1.0, 1.0])
    voxel_size = (max_bound - min_bound) / resolution
    
    def get_voxels(points):
        indices = np.floor((points - min_bound) / voxel_size).astype(int)
        indices = np.clip(indices, 0, resolution - 1)
        return set(map(tuple, indices))

    voxels_1 = get_voxels(point_cloud_1)
    voxels_2 = get_voxels(point_cloud_2)
    
    intersection = len(voxels_1.intersection(voxels_2))
    denom = len(voxels_1) + len(voxels_2)
    
    return (2.0 * intersection) / denom if denom > 0 else 0.0

def compute_chamfer_distance(pred_points, gt_points):
    """Computes Chamfer Distance (CD)."""
    if len(pred_points) == 0 or len(gt_points) == 0: return float('inf')
    
    gt_tree = cKDTree(gt_points)
    pred_tree = cKDTree(pred_points)
    
    dist_pred_to_gt, _ = gt_tree.query(pred_points, k=1)
    dist_gt_to_pred, _ = pred_tree.query(gt_points, k=1)
    
    return np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)

def compute_emd(pred_points, gt_points, max_points=2048):
    """Computes Earth Mover's Distance (EMD)."""
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float('inf')

    n_points = min(len(pred_points), len(gt_points), max_points)
    
    np.random.seed(42)
    indices_pred = np.random.choice(len(pred_points), n_points, replace=False)
    indices_gt = np.random.choice(len(gt_points), n_points, replace=False)
    
    pred_sampled = pred_points[indices_pred]
    gt_sampled = gt_points[indices_gt]
    
    dists = cdist(pred_sampled, gt_sampled, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(dists)
    
    total_cost = dists[row_ind, col_ind].sum()
    emd_value = total_cost / n_points
    
    return emd_value

# ==========================================
# 2. Data Extraction Helpers (Updated for NII)
# ==========================================

def load_and_sample_ply(ply_input, num_points=10000):
    """Loads a PLY file and samples points."""
    if isinstance(ply_input, str):
        if not os.path.exists(ply_input):
            print(f"Error: PLY file not found at {ply_input}")
            return np.zeros((0, 3))
        try:
            mesh = o3d.io.read_triangle_mesh(ply_input)
            if len(mesh.triangles) > 0:
                pcd = mesh.sample_points_uniformly(number_of_points=num_points)
                return np.asarray(pcd.points)
            else:
                pcd = o3d.io.read_point_cloud(ply_input)
                if len(pcd.points) > num_points:
                    ratio = num_points / len(pcd.points)
                    pcd = pcd.random_down_sample(ratio)
                return np.asarray(pcd.points)
        except Exception as e:
            print(f"PLY Loading Error: {e}")
            return np.zeros((0, 3))
    return np.zeros((0, 3))

def load_nii_as_surface_points(nii_path, num_points=10000, iso_value=0.5):
    """
    Loads a NIfTI file, extracts the isosurface (Marching Cubes), 
    and samples points from that surface.
    """
    if not os.path.exists(nii_path):
        print(f"Error: NIfTI file not found at {nii_path}")
        return np.zeros((0, 3))
    
    try:
        # 1. Load NIfTI
        nii = nib.load(nii_path)
        data = nii.get_fdata()
        
        # 2. Check if empty
        if data.max() < iso_value:
            print("Warning: NIfTI volume is empty (all values below threshold).")
            return np.zeros((0, 3))

        # 3. Marching Cubes to get surface vertices (in index coordinates)
        # We assume the mask is binary or has a clear threshold
        verts, faces, _, _ = measure.marching_cubes(data, level=iso_value)
        
        # 4. Apply voxel spacing (zooms) to preserve aspect ratio
        # We do NOT apply full affine (translation/rotation) usually, because
        # we are going to normalize to unit cube anyway. But scaling matters.
        zooms = nii.header.get_zooms()[:3]
        verts = verts * np.array(zooms)
        
        # 5. Convert to Open3D Mesh for easy sampling
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        
        # 6. Sample Points
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        return np.asarray(pcd.points)

    except Exception as e:
        print(f"NIfTI Processing Error: {e}")
        return np.zeros((0, 3))

# ==========================================
# 3. Main Comparison Function (PLY vs NII)
# ==========================================

def evaluate_ply_vs_nii(pred_ply_path, gt_nii_path, num_points=10000, do_icp=True):
    """
    Compares a generated PLY against a Ground Truth NIfTI.
    Returns: Dictionary of metrics or None if failed.
    """
    print(f"\n--- Comparing Prediction (PLY) vs GT (NIfTI) ---")
    print(f"Pred: {os.path.basename(pred_ply_path)}")
    print(f"GT:   {os.path.basename(gt_nii_path)}")
    
    # 1. Extraction
    print("Loading Prediction PLY...")
    pred_points = load_and_sample_ply(pred_ply_path, num_points=num_points)
    
    print("Extracting Surface from GT NIfTI...")
    gt_points = load_nii_as_surface_points(gt_nii_path, num_points=num_points)
    
    # Handle Missing/Empty
    if len(pred_points) == 0 or len(gt_points) == 0:
        print("Error: One of the inputs yielded 0 points.")
        return None
        
    print(f"Loaded points -> Pred: {len(pred_points)}, GT (Surface): {len(gt_points)}")

    # 2. Normalize 
    gt_norm = normalize_to_unit_cube(gt_points)
    pred_norm = normalize_to_unit_cube(pred_points)
    
    # 3. Align (ICP)
    if do_icp:
        print("Applying ICP alignment...")
        pred_aligned = apply_icp(pred_norm, gt_norm, threshold=0.3)
    else:
        pred_aligned = pred_norm
    
    # 4. Compute Metrics
    print("Computing Metrics...")
    f1, precision, recall = compute_f1_score(pred_aligned, gt_norm)
    iou = compute_voxel_iou(pred_aligned, gt_norm)
    dice = compute_voxel_dice(pred_aligned, gt_norm)
    chamfer = compute_chamfer_distance(pred_aligned, gt_norm)
    emd = compute_emd(pred_aligned, gt_norm, max_points=2048)
    
    print("-" * 30)
    print(f"Results: {os.path.basename(pred_ply_path)}")
    print(f"  Dice: {dice:.4f}, CD: {chamfer:.4f}")
    print("-" * 30)
    
    return {
        "File_ID": os.path.basename(pred_ply_path),
        "F1_Score": f1,
        "Precision": precision,
        "Recall": recall,
        "Voxel_IoU": iou,
        "Voxel_Dice": dice,
        "Chamfer_Dist": chamfer,
        "EMD": emd
    }

# --- Example Usage ---
if __name__ == "__main__":
    # Update these paths to your actual files
    # pred_ply_folder = '/PHShome/yl535/project/python/sam_3d/sam-3d-objects/results_AeroPath/lungs/'
    # gt_nii_folder   = '/PHShome/yl535/project/python/datasets/AeroPath/lungs_segmented/'
    # output_csv_path = os.path.join(os.path.dirname(pred_ply_folder), f'perf_{os.path.basename(pred_ply_folder)}.csv')
    
    parser = argparse.ArgumentParser(description="Evaluate PLY predictions against NIfTI Ground Truth.")
    
    parser.add_argument(
        '--pred_folder', 
        type=str, 
        required=True,
        help='Path to the folder containing prediction PLY files.'
    )
    parser.add_argument(
        '--gt_folder', 
        type=str, 
        required=True,
        help='Path to the folder containing Ground Truth NIfTI files.'
    )
    parser.add_argument(
        '--file_suffix', 
        type=str, 
        default='.nii.gz',
        help='Suffix for CT files corresponding to the mask files.'
    )
    
    args = parser.parse_args()

    pred_ply_folder = args.pred_folder
    gt_nii_folder = args.gt_folder
    output_csv_path = os.path.join(os.path.dirname(pred_ply_folder), f'perf_{os.path.basename(pred_ply_folder)}.csv')


    p = Path(gt_nii_folder)
    files = list(p.rglob(f'*_mask{args.file_suffix}'))
    
    # List to store results dictionaries
    all_metrics = []

    for file_path in files:
        # 1. Get the full basename (e.g., "subject01_mask.nii.gz")
        mask_name = file_path.name
        ct_name = mask_name.replace(f'_mask{args.file_suffix}', f'{args.file_suffix}.ply')
        
        pred_ply = os.path.join(pred_ply_folder, ct_name)
        gt_nii = os.path.join(gt_nii_folder, mask_name)
        
        # 2. Get metrics dictionary
        metrics_dict = evaluate_ply_vs_nii(pred_ply, gt_nii)
        
        # 3. Append if successful
        if metrics_dict is not None:
            all_metrics.append(metrics_dict)
    
    # 4. Aggregate and Save Statistics
    if len(all_metrics) > 0:
        df = pd.DataFrame(all_metrics)
        
        # Compute Mean and Std (excluding the filename column)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        mean_row = df[numeric_cols].mean()
        std_row = df[numeric_cols].std()
        
        # Convert Series to DataFrame to append
        summary_stats = pd.DataFrame([mean_row, std_row], index=['Mean', 'Std'])
        
        # Rename the 'File_ID' column for the summary rows to be descriptive
        summary_stats['File_ID'] = ['Average', 'Standard Deviation']
        
        # Concatenate original results with summary stats at the bottom
        final_df = pd.concat([df, summary_stats], ignore_index=False)
        
        # Save to CSV
        final_df.to_csv(output_csv_path, index=False)
        print(f"\nProcessing complete. Metrics saved to {output_csv_path}")
        print("\nSummary Statistics:")
        print(summary_stats)
    else:
        print("\nNo valid comparisons were processed.")