"""
Utilities package for Zero-Shot-SAM3Dfy.

Modules:
- metrics: 3D reconstruction evaluation metrics (F-score, Chamfer, IoU, etc.)
- preprocessing: Image preparation for 3D reconstruction models
- visualization: GIF creation and plotting for meshes and point clouds
- tools: Mesh and point cloud utility functions
"""

from .metrics import (
    evaluate_reconstruction,
    compute_fscore,
    compute_chamfer_distance,
    compute_voxel_iou,
    compute_emd,
    compute_precision_recall,
    compute_voxel_dice,
    icp_align,
    normalize_point_clouds,
    print_metrics,
)

from .preprocessing import (
    prepare_slice_for_hunyuan,
    normalize_to_unit_range,
)

from .visualization import (
    create_mesh_rotation_gif,
    create_comparison_gif,
    plot_slice_with_mask,
    points_to_mesh,
    render_mesh_on_axis,
)

from .tools import (
    load_mesh,
    save_mesh,
    mesh_to_pointcloud,
    normalize_points,
    get_mesh_vertices,
    get_mesh_faces,
    compute_mesh_bounds,
    subsample_points,
)

__all__ = [
    # metrics
    'evaluate_reconstruction',
    'compute_fscore',
    'compute_chamfer_distance',
    'compute_voxel_iou',
    'compute_emd',
    'compute_precision_recall',
    'compute_voxel_dice',
    'icp_align',
    'normalize_point_clouds',
    'print_metrics',
    # preprocessing
    'prepare_slice_for_hunyuan',
    'normalize_to_unit_range',
    # visualization
    'create_mesh_rotation_gif',
    'create_comparison_gif',
    'plot_slice_with_mask',
    'points_to_mesh',
    'render_mesh_on_axis',
    # tools
    'load_mesh',
    'save_mesh',
    'mesh_to_pointcloud',
    'normalize_points',
    'get_mesh_vertices',
    'get_mesh_faces',
    'compute_mesh_bounds',
    'subsample_points',
]
