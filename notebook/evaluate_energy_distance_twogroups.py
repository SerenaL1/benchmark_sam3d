import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import os
import glob
import time

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
    # (Symmetric Chamfer: mean(d(p1->p2)) + mean(d(p2->p1)))
    results['chamfer'] = np.mean(dist_p1_to_p2) + np.mean(dist_p2_to_p1)

    # --- EMD ---
    n_points = min(len(p1), len(p2), max_emd_points)
    # Simple random sampling for EMD speed
    # Note: For strict determinism, set seed inside or pass seed
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
        # Try mesh first
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.triangles) > 0:
            pcd = mesh.sample_points_uniformly(number_of_points=num_points)
            points = np.asarray(pcd.points)
        else:
            pcd = o3d.io.read_point_cloud(path)
            # Basic fallback sampling
            if len(pcd.points) > 0:
                if len(pcd.points) > num_points:
                    pcd = pcd.random_down_sample(num_points / len(pcd.points))
                points = np.asarray(pcd.points)
            else:
                return None

        return normalize_to_unit_cube(points)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return None

def load_group(file_paths, num_points=10000):
    """Loads a list of file paths into memory as normalized numpy arrays."""
    data = []
    print(f"Loading {len(file_paths)} files...")
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
    Computes Energy Distance between two groups of shapes (Group A and Group B).
    
    Energy Distance D^2(A, B) = 2*E[d(a,b)] - E[d(a,a')] - E[d(b,b')]
    
    For similarity metrics (F1, IoU, Dice), we convert to distance: d = 1 - score.
    For distance metrics (Chamfer, EMD), we use raw values.
    """
    N = len(group_A)
    M = len(group_B)
    
    if N == 0 or M == 0:
        print("Error: One of the groups is empty.")
        return None

    # Initialize accumulation matrices
    # We will store: sum, count to calculate means later
    keys = ['f1', 'iou', 'dice', 'chamfer', 'emd']
    
    # Structures to hold raw distances for each component of Energy Distance
    # d_XY represents distances between Group A and Group B
    # d_XX represents distances within Group A
    # d_YY represents distances within Group B
    d_XY = {k: [] for k in keys}
    d_XX = {k: [] for k in keys}
    d_YY = {k: [] for k in keys}

    print("\n--- Computing Cross-Distances (A vs B) ---")
    # 1. Compute Cross-Term (A vs B)
    # We compute all N*M pairs
    count = 0
    total_pairs = N * M
    for i, p_a in enumerate(group_A):
        for j, p_b in enumerate(group_B):
            if count % 10 == 0: print(f"  Processing pair {count}/{total_pairs}...", end='\r')
            
            # Align A to B (or vice versa)
            if do_icp:
                p_a_aligned = apply_icp(p_a, p_b)
            else:
                p_a_aligned = p_a
                
            res = compute_metrics_pair(p_a_aligned, p_b)
            
            # Store values
            # Convert Similarity -> Distance
            d_XY['f1'].append(1.0 - res['f1'])
            d_XY['iou'].append(1.0 - res['iou'])
            d_XY['dice'].append(1.0 - res['dice'])
            # Keep Distances as is
            d_XY['chamfer'].append(res['chamfer'])
            d_XY['emd'].append(res['emd'])
            count += 1

    print("\n--- Computing Self-Distances (A vs A) ---")
    # 2. Compute Self-Term (A vs A)
    # We only need upper triangle for distinct pairs, but Energy Distance usually 
    # implies full sum. d(x,x) is 0.
    count = 0
    # Optimization: Only loop i < j (since dist(a,b) ~ dist(b,a)) 
    # and assume dist(a,a) = 0.
    # However, strictly speaking, ICP is not perfectly symmetric. 
    # For rigorousness, we do full loops or assume symmetry. 
    # Let's assume symmetry to save 50% time: d(i,j) == d(j,i).
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
            
            # Add twice (for i,j and j,i)
            d_XX['f1'].extend([val_f1, val_f1])
            d_XX['iou'].extend([val_iou, val_iou])
            d_XX['dice'].extend([val_dice, val_dice])
            d_XX['chamfer'].extend([val_cd, val_cd])
            d_XX['emd'].extend([val_emd, val_emd])
    
    # Add zeros for the diagonal (i=i) if we want technically correct means over N*N
    # N zeros for each metric
    for k in keys:
        d_XX[k].extend([0.0] * N)

    print("\n--- Computing Self-Distances (B vs B) ---")
    # 3. Compute Self-Term (B vs B)
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

    # 4. Final Calculation
    energy_results = {}
    print("\n" + "="*40)
    print(f"ENERGY DISTANCE RESULTS (A: {N} files, B: {M} files)")
    print("Interpretation: Lower is more similar distributions.")
    print("For F1/IoU/Dice, metric used was (1 - Score).")
    print("="*40)

    for k in keys:
        # Calculate Expectation (Mean)
        E_XY = np.mean(d_XY[k])
        E_XX = np.mean(d_XX[k])
        E_YY = np.mean(d_YY[k])
        
        # Energy Distance Formula
        ed_value = 2 * E_XY - E_XX - E_YY
        
        energy_results[k] = ed_value
        print(f"{k.upper():<10} | Energy Dist: {ed_value:.6f} (Avg Cross: {E_XY:.4f}, Avg Intra-A: {E_XX:.4f}, Avg Intra-B: {E_YY:.4f})")

    return energy_results

def organize_ply_files(folder_path):
    """
    Iterates through a folder and sorts .ply files into lists based on 
    their suffix (_A, _D, _G, _N).
    
    Args:
        folder_path (str): The directory containing the .ply files.
        
    Returns:
        dict: A dictionary with keys 'A', 'D', 'G', 'N', where values 
              are lists of full file paths.
    """
    
    # Initialize the container for your lists
    ply_groups = {
        'A': [],
        'D': [],
        'G': [],
        'N': []
    }
    
    # Verify directory exists
    if not os.path.exists(folder_path):
        print(f"Error: Directory '{folder_path}' not found.")
        return ply_groups

    # Iterate over all files in the directory
    for filename in os.listdir(folder_path):
        # We only care about .ply files
        if not filename.endswith(".ply"):
            continue
            
        # Construct full path (useful for loading later)
        full_path = os.path.join(folder_path, filename)
        
        # Check suffixes and append to the correct list
        # We use .endswith() to ensure we match the file structure accurately
        if filename.endswith("_A.ply"):
            ply_groups['A'].append(full_path)
        elif filename.endswith("_D.ply"):
            ply_groups['D'].append(full_path)
        elif filename.endswith("_G.ply"):
            ply_groups['G'].append(full_path)
        elif filename.endswith("_N.ply"):
            ply_groups['N'].append(full_path)
        else:
            # Optional: Print files that didn't match the expected pattern
            # print(f"Skipped file (unknown format): {filename}")
            pass

    return ply_groups

# ==========================================
# 4. Main Execution
# ==========================================

if __name__ == "__main__":
    input_folder = "/PHShome/yl535/project/python/sam_3d/sam-3d-objects/results/FIVES/gaussians"
    
    categorized_files = organize_ply_files(input_folder)

    list_A = categorized_files['A']
    list_D = categorized_files['D']
    list_G = categorized_files['G']
    list_N = categorized_files['N']
    
    data_A = load_group(list_N)
    data_B = load_group(list_A)

    # Run Evaluation
    if len(data_A) > 0 and len(data_B) > 0:
        start_time = time.time()
        results = calculate_energy_distance(data_A, data_B, do_icp=True)
        print(f"\nTotal calculation time: {time.time() - start_time:.2f} seconds")
    else:
        print("No data loaded.")

    data_A = load_group(list_N)
    data_B = load_group(list_D)

    # Run Evaluation
    if len(data_A) > 0 and len(data_B) > 0:
        start_time = time.time()
        results = calculate_energy_distance(data_A, data_B, do_icp=True)
        print(f"\nTotal calculation time: {time.time() - start_time:.2f} seconds")
    else:
        print("No data loaded.")

    data_A = load_group(list_N)
    data_B = load_group(list_G)

    # Run Evaluation
    if len(data_A) > 0 and len(data_B) > 0:
        start_time = time.time()
        results = calculate_energy_distance(data_A, data_B, do_icp=True)
        print(f"\nTotal calculation time: {time.time() - start_time:.2f} seconds")
    else:
        print("No data loaded.")