import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import os
import glob
import time
import logging

# ==========================================
# 0. Logging Setup
# ==========================================
def setup_logger(output_dir="."):
    """Sets up a logger that writes to both console and a file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = os.path.join(output_dir, f"energy_distance_log_{timestamp}.txt")
    
    # Create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers if any (to prevent duplicates in notebooks/re-runs)
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(message)s') # Simple format for results
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging started. Saving output to: {log_filename}")
    return logger

# Initialize logger as global for convenience in this script
# We will actually configure it in __main__ to set the path correctly
logger = logging.getLogger()

# ==========================================
# 1. Metric Helper Functions
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
    # Using point-to-point for general robustness without normals
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    source_pcd.transform(reg_p2p.transformation)
    return np.asarray(source_pcd.points)

def compute_metrics_pair(p1, p2, threshold=0.01, resolution=64, max_emd_points=2048):
    """
    Computes all 5 metrics for a single pair of point clouds (normalized).
    Returns a dictionary of results.
    """
    results = {}
    
    # --- F1 Score ---
    gt_tree = cKDTree(p2)
    pred_tree = cKDTree(p1)
    
    dist_p1_to_p2, _ = gt_tree.query(p1, k=1)
    precision = np.mean(dist_p1_to_p2 < threshold)
    
    dist_p2_to_p1, _ = pred_tree.query(p2, k=1)
    recall = np.mean(dist_p2_to_p1 < threshold)
    
    if precision + recall == 0:
        results['f1'] = 0.0
    else:
        results['f1'] = 2 * (precision * recall) / (precision + recall)

    # --- Voxel IoU & Dice ---
    min_bound = np.array([-1.0, -1.0, -1.0])
    voxel_size = 2.0 / resolution # range [-1, 1] is size 2
    
    def get_voxels(points):
        indices = np.floor((points - min_bound) / voxel_size).astype(int)
        indices = np.clip(indices, 0, resolution - 1)
        return set(map(tuple, indices))

    voxels_1 = get_voxels(p1)
    voxels_2 = get_voxels(p2)
    
    intersection = len(voxels_1.intersection(voxels_2))
    union = len(voxels_1.union(voxels_2))
    denom = len(voxels_1) + len(voxels_2)
    
    results['iou'] = intersection / union if union > 0 else 0.0
    results['dice'] = (2.0 * intersection) / denom if denom > 0 else 0.0

    # --- Chamfer Distance ---
    results['chamfer'] = np.mean(dist_p1_to_p2) + np.mean(dist_p2_to_p1)

    # --- EMD ---
    n_points = min(len(p1), len(p2), max_emd_points)
    idx1 = np.random.choice(len(p1), n_points, replace=False)
    idx2 = np.random.choice(len(p2), n_points, replace=False)
    
    dists = cdist(p1[idx1], p2[idx2], metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(dists)
    results['emd'] = dists[row_ind, col_ind].sum() / n_points

    return results

# ==========================================
# 2. Data Loading Helpers
# ==========================================

def load_and_process_ply(path, num_points=10000):
    """Loads a PLY, samples it, and normalizes it to unit cube."""
    try:
        # Load Mesh or PCD
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.triangles) > 0:
            pcd = mesh.sample_points_uniformly(number_of_points=num_points)
            points = np.asarray(pcd.points)
        else:
            pcd = o3d.io.read_point_cloud(path)
            if len(pcd.points) > 0:
                if len(pcd.points) > num_points:
                    pcd = pcd.random_down_sample(num_points / len(pcd.points))
                points = np.asarray(pcd.points)
            else:
                return None

        return normalize_to_unit_cube(points)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return None

def load_group(file_paths, num_points=10000):
    """Loads a list of file paths into memory as normalized numpy arrays."""
    data = []
    # Use logger.info for permanent record, print for transient status if needed
    logger.info(f"Loading {len(file_paths)} files...") 
    for f in file_paths:
        pts = load_and_process_ply(f, num_points)
        if pts is not None and len(pts) > 0:
            data.append(pts)
    return data

# ==========================================
# 3. Energy Distance Logic
# ==========================================

def calculate_energy_distance(group_A, group_B, do_icp=True):
    """
    Computes Energy Distance between two groups of shapes.
    Uses logger for main outputs and print() for progress bars (to keep log clean).
    """
    N = len(group_A)
    M = len(group_B)
    
    if N == 0 or M == 0:
        logger.error("Error: One of the groups is empty.")
        return None

    keys = ['f1', 'iou', 'dice', 'chamfer', 'emd']
    d_XY = {k: [] for k in keys}
    d_XX = {k: [] for k in keys}
    d_YY = {k: [] for k in keys}

    logger.info("\n--- Computing Cross-Distances (A vs B) ---")
    count = 0
    total_pairs = N * M
    for i, p_a in enumerate(group_A):
        for j, p_b in enumerate(group_B):
            # Print to console only (end='\r'), do not log progress to file
            if count % 5 == 0: 
                print(f"  Processing pair {count}/{total_pairs}...", end='\r')
            
            if do_icp:
                p_a_aligned = apply_icp(p_a, p_b)
            else:
                p_a_aligned = p_a
                
            res = compute_metrics_pair(p_a_aligned, p_b)
            
            d_XY['f1'].append(1.0 - res['f1'])
            d_XY['iou'].append(1.0 - res['iou'])
            d_XY['dice'].append(1.0 - res['dice'])
            d_XY['chamfer'].append(res['chamfer'])
            d_XY['emd'].append(res['emd'])
            count += 1
    print(" " * 50, end='\r') # Clear progress line on console

    logger.info("--- Computing Self-Distances (A vs A) ---")
    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            if do_icp:
                p_i_aligned = apply_icp(group_A[i], group_A[j])
            else:
                p_i_aligned = group_A[i]

            res = compute_metrics_pair(p_i_aligned, group_A[j])
            
            val_f1 = 1.0 - res['f1']
            val_iou = 1.0 - res['iou']
            val_dice = 1.0 - res['dice']
            val_cd = res['chamfer']
            val_emd = res['emd']
            
            d_XX['f1'].extend([val_f1, val_f1])
            d_XX['iou'].extend([val_iou, val_iou])
            d_XX['dice'].extend([val_dice, val_dice])
            d_XX['chamfer'].extend([val_cd, val_cd])
            d_XX['emd'].extend([val_emd, val_emd])
    
    for k in keys:
        d_XX[k].extend([0.0] * N)

    logger.info("--- Computing Self-Distances (B vs B) ---")
    for i in range(M):
        for j in range(i + 1, M):
            if do_icp:
                p_i_aligned = apply_icp(group_B[i], group_B[j])
            else:
                p_i_aligned = group_B[i]

            res = compute_metrics_pair(p_i_aligned, group_B[j])
            
            val_f1 = 1.0 - res['f1']
            val_iou = 1.0 - res['iou']
            val_dice = 1.0 - res['dice']
            val_cd = res['chamfer']
            val_emd = res['emd']
            
            d_YY['f1'].extend([val_f1, val_f1])
            d_YY['iou'].extend([val_iou, val_iou])
            d_YY['dice'].extend([val_dice, val_dice])
            d_YY['chamfer'].extend([val_cd, val_cd])
            d_YY['emd'].extend([val_emd, val_emd])
            
    for k in keys:
        d_YY[k].extend([0.0] * M)

    energy_results = {}
    logger.info("\n" + "="*40)
    logger.info(f"ENERGY DISTANCE RESULTS (A: {N} files, B: {M} files)")
    logger.info("Interpretation: Lower is more similar distributions.")
    logger.info("="*40)

    for k in keys:
        E_XY = np.mean(d_XY[k])
        E_XX = np.mean(d_XX[k])
        E_YY = np.mean(d_YY[k])
        
        ed_value = 2 * E_XY - E_XX - E_YY
        energy_results[k] = ed_value
        logger.info(f"{k.upper():<10} | Energy Dist: {ed_value:.6f} (Avg Cross: {E_XY:.4f}, Avg Intra-A: {E_XX:.4f}, Avg Intra-B: {E_YY:.4f})")

    return energy_results

def organize_ply_files(folder_path):
    """Sorts .ply files into lists based on suffix (_A, _D, _G, _N)."""
    ply_groups = {'A': [], 'D': [], 'G': [], 'N': []}
    
    if not os.path.exists(folder_path):
        logger.error(f"Error: Directory '{folder_path}' not found.")
        return ply_groups

    for filename in os.listdir(folder_path):
        if not filename.endswith(".ply"):
            continue
        
        full_path = os.path.join(folder_path, filename)
        
        if filename.endswith("_A.ply"):
            ply_groups['A'].append(full_path)
        elif filename.endswith("_D.ply"):
            ply_groups['D'].append(full_path)
        elif filename.endswith("_G.ply"):
            ply_groups['G'].append(full_path)
        elif filename.endswith("_N.ply"):
            ply_groups['N'].append(full_path)

    return ply_groups

# ==========================================
# 4. Main Execution
# ==========================================

if __name__ == "__main__":
    input_folder = "/PHShome/yl535/project/python/sam_3d/sam-3d-objects/results/FIVES/gaussians"
    
    # 1. Initialize Logger (output will be in the same folder as input, or customize)
    # Saving log to the input folder:
    setup_logger(output_dir=os.path.dirname(input_folder))
    
    categorized_files = organize_ply_files(input_folder)

    list_A = categorized_files['A']
    list_D = categorized_files['D']
    list_G = categorized_files['G']
    list_N = categorized_files['N']
    
    # --- Comparison 1: N vs A ---
    logger.info("\n" + "#"*50)
    logger.info("COMPARISON 1: Group N vs Group A")
    logger.info("#"*50)
    
    data_A_N = load_group(list_N)
    data_B_A = load_group(list_A)

    if len(data_A_N) > 0 and len(data_B_A) > 0:
        start_time = time.time()
        results = calculate_energy_distance(data_A_N, data_B_A, do_icp=True)
        logger.info(f"\nTotal calculation time: {time.time() - start_time:.2f} seconds")
    else:
        logger.info("No data loaded for N vs A.")

    # --- Comparison 2: N vs D ---
    logger.info("\n" + "#"*50)
    logger.info("COMPARISON 2: Group N vs Group D")
    logger.info("#"*50)
    
    # Reloading isn't strictly necessary if N is the same, but kept for safety/independence
    data_A_N = load_group(list_N) 
    data_B_D = load_group(list_D)

    if len(data_A_N) > 0 and len(data_B_D) > 0:
        start_time = time.time()
        results = calculate_energy_distance(data_A_N, data_B_D, do_icp=True)
        logger.info(f"\nTotal calculation time: {time.time() - start_time:.2f} seconds")
    else:
        logger.info("No data loaded for N vs D.")

    # --- Comparison 3: N vs G ---
    logger.info("\n" + "#"*50)
    logger.info("COMPARISON 3: Group N vs Group G")
    logger.info("#"*50)

    data_A_N = load_group(list_N)
    data_B_G = load_group(list_G)

    if len(data_A_N) > 0 and len(data_B_G) > 0:
        start_time = time.time()
        results = calculate_energy_distance(data_A_N, data_B_G, do_icp=True)
        logger.info(f"\nTotal calculation time: {time.time() - start_time:.2f} seconds")
    else:
        logger.info("No data loaded for N vs G.")