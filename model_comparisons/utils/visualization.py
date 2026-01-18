"""
Visualization utilities for 3D reconstruction.

Functions for creating GIFs, plots, and comparisons of meshes and point clouds.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio
import trimesh
from scipy.spatial import Delaunay
from typing import Union, Optional
from pathlib import Path


def points_to_mesh(points: np.ndarray, alpha: float = 0.3) -> Optional[trimesh.Trimesh]:
    """
    Convert a point cloud to a mesh using alpha shape / convex hull approach.
    Uses ball pivoting or convex hull for simple mesh generation.

    Args:
        points: Point cloud (N, 3)
        alpha: Alpha value for alpha shape (smaller = tighter fit)

    Returns:
        Trimesh object or None if failed
    """
    try:
        # Use trimesh's convex hull as a simple approach
        cloud = trimesh.PointCloud(points)
        mesh = cloud.convex_hull
        return mesh
    except Exception:
        return None


def render_mesh_on_axis(
    ax,
    vertices: np.ndarray,
    faces: np.ndarray,
    base_color: tuple = (0.7, 0.7, 0.7)
):
    """
    Render a mesh with lighting on a matplotlib 3D axis.

    Args:
        ax: Matplotlib 3D axis
        vertices: Mesh vertices (N, 3)
        faces: Mesh faces (F, 3)
        base_color: RGB base color tuple
    """
    mesh_collection = Poly3DCollection(
        vertices[faces],
        alpha=1.0,
        facecolor='gray',
        edgecolor='none',
        linewidth=0
    )

    # Compute face normals for shading
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]]
    )
    face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8)

    # Simple directional lighting
    light_dir = np.array([0.5, 0.5, 1.0])
    light_dir = light_dir / np.linalg.norm(light_dir)
    intensities = np.abs(np.dot(face_normals, light_dir))
    intensities = 0.3 + 0.7 * intensities  # ambient + diffuse

    # Apply shading colors
    colors = np.zeros((len(faces), 4))
    colors[:, 0] = intensities * base_color[0]
    colors[:, 1] = intensities * base_color[1]
    colors[:, 2] = intensities * base_color[2]
    colors[:, 3] = 1.0
    mesh_collection.set_facecolors(colors)

    ax.add_collection3d(mesh_collection)


def create_mesh_rotation_gif(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    output_path: Union[str, Path],
    sample_name: str = "",
    n_frames: int = 36,
    duration: float = 0.1,
    figsize: tuple = (8, 8)
) -> str:
    """
    Create a rotating GIF of a 3D mesh with proper solid rendering and lighting.

    Args:
        mesh: Trimesh mesh or Scene object
        output_path: Path to save the GIF
        sample_name: Name to display as title
        n_frames: Number of frames in the animation
        duration: Duration of each frame in seconds
        figsize: Figure size

    Returns:
        Path to saved GIF
    """
    # Handle Scene objects
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()

    vertices = mesh.vertices
    faces = mesh.faces

    # Normalize mesh to [-1, 1]
    center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2
    scale = (vertices.max(axis=0) - vertices.min(axis=0)).max() / 2
    if scale == 0:
        scale = 1.0
    vertices_norm = (vertices - center) / scale

    frames = []

    for i in range(n_frames):
        angle = 360 * i / n_frames

        fig = plt.figure(figsize=figsize, facecolor='black')
        ax = fig.add_subplot(111, projection='3d', facecolor='black')

        render_mesh_on_axis(ax, vertices_norm, faces)

        # Set view
        ax.view_init(elev=20, azim=angle)

        # Set axis limits
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])

        # Hide axes for cleaner look
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])

        if sample_name:
            ax.set_title(sample_name, color='white', fontsize=12)

        plt.tight_layout()

        # Convert figure to image
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf)[:, :, :3]
        frames.append(image.copy())
        plt.close(fig)

    imageio.mimsave(str(output_path), frames, duration=duration, loop=0)
    return str(output_path)


def render_pointcloud_with_shading(
    ax,
    points: np.ndarray,
    base_color: tuple = (0.7, 0.7, 0.7),
    point_size: float = 0.5
):
    """
    Render a point cloud with depth-based shading on a matplotlib 3D axis.

    Args:
        ax: Matplotlib 3D axis
        points: Point cloud (N, 3)
        base_color: RGB base color tuple
        point_size: Size of points
    """
    # Compute pseudo-normals using local PCA or simple depth shading
    # Use z-coordinate for simple depth-based shading
    z_vals = points[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    if z_max - z_min > 1e-8:
        intensities = 0.4 + 0.6 * (z_vals - z_min) / (z_max - z_min)
    else:
        intensities = np.ones(len(points)) * 0.7

    # Create grayscale colors with shading
    colors = np.zeros((len(points), 3))
    colors[:, 0] = intensities * base_color[0]
    colors[:, 1] = intensities * base_color[1]
    colors[:, 2] = intensities * base_color[2]

    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colors, s=point_size, alpha=0.9
    )


def create_comparison_gif(
    gt_mesh_or_points: Union[np.ndarray, trimesh.Trimesh, trimesh.Scene],
    pred_mesh_or_points: Union[np.ndarray, trimesh.Trimesh, trimesh.Scene],
    output_path: Union[str, Path],
    sample_name: str = "",
    n_frames: int = 36,
    duration: float = 0.1,
    use_icp_align: bool = True,
    render_mesh: bool = True,
    gt_color: tuple = (0.7, 0.7, 0.7),  # Gray
    pred_color: tuple = (0.7, 0.7, 0.7),  # Gray
    pred_rotation: Optional[tuple] = None  # (axis, angle_degrees) for fixing orientation
) -> str:
    """
    Create a side-by-side rotating GIF comparing GT and predicted meshes.
    Both GT and predicted are rendered as solid meshes with lighting.
    Uses ICP alignment to align predicted to GT. All points are used (no subsampling).

    Args:
        gt_mesh_or_points: Ground truth mesh or point cloud (pre-normalized to [-1, 1])
        pred_mesh_or_points: Predicted mesh or point cloud
        output_path: Path to save the GIF
        sample_name: Name to display as title
        n_frames: Number of frames
        duration: Duration per frame
        use_icp_align: Whether to ICP align predicted to GT
        render_mesh: If True, render as solid mesh; if False, render as point cloud
        gt_color: RGB color for GT (default gray)
        pred_color: RGB color for predicted (default gray)
        pred_rotation: Tuple of (axis, angle_degrees) to rotate predicted mesh before display.
                       axis is 'x', 'y', or 'z'. E.g., ('x', 90) rotates 90 degrees around x-axis.

    Returns:
        Path to saved GIF
    """
    from .metrics import icp_align
    from scipy.spatial.transform import Rotation as R

    # Process GT mesh/points
    gt_vertices = None
    gt_faces = None
    gt_is_mesh = False

    if isinstance(gt_mesh_or_points, (trimesh.Trimesh, trimesh.Scene)):
        if isinstance(gt_mesh_or_points, trimesh.Scene):
            gt_mesh = gt_mesh_or_points.to_geometry()
        else:
            gt_mesh = gt_mesh_or_points
        gt_vertices = np.array(gt_mesh.vertices)
        gt_faces = np.array(gt_mesh.faces)
        gt_is_mesh = True
    elif gt_mesh_or_points is not None and len(gt_mesh_or_points) > 0:
        # It's a point cloud - convert to mesh (use all points)
        gt_points = gt_mesh_or_points

        # Create mesh from point cloud using convex hull
        gt_mesh = points_to_mesh(gt_points)
        if gt_mesh is not None:
            gt_vertices = np.array(gt_mesh.vertices)
            gt_faces = np.array(gt_mesh.faces)
            gt_is_mesh = True
        else:
            # Fallback to point cloud
            gt_vertices = gt_points
            gt_faces = None
            gt_is_mesh = False

    # Process predicted mesh/points
    pred_vertices = None
    pred_faces = None
    pred_is_mesh = False

    if isinstance(pred_mesh_or_points, (trimesh.Trimesh, trimesh.Scene)):
        if isinstance(pred_mesh_or_points, trimesh.Scene):
            pred_mesh = pred_mesh_or_points.to_geometry()
        else:
            pred_mesh = pred_mesh_or_points

        pred_vertices = np.array(pred_mesh.vertices)
        pred_faces = np.array(pred_mesh.faces)

        # Normalize predicted mesh to [-1, 1]
        center = (pred_vertices.max(axis=0) + pred_vertices.min(axis=0)) / 2
        scale = (pred_vertices.max(axis=0) - pred_vertices.min(axis=0)).max() / 2
        if scale == 0:
            scale = 1.0
        pred_vertices = (pred_vertices - center) / scale
        pred_is_mesh = True
    elif pred_mesh_or_points is not None and len(pred_mesh_or_points) > 0:
        pred_vertices = pred_mesh_or_points
        pred_faces = None
        pred_is_mesh = False

    # Apply rotation to predicted vertices if specified
    if pred_rotation is not None and pred_vertices is not None:
        axis, angle_deg = pred_rotation
        angle_rad = np.deg2rad(angle_deg)
        if axis == 'x':
            rot = R.from_euler('x', angle_rad)
        elif axis == 'y':
            rot = R.from_euler('y', angle_rad)
        elif axis == 'z':
            rot = R.from_euler('z', angle_rad)
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'.")
        pred_vertices = rot.apply(pred_vertices)

    # ICP align predicted to GT
    if use_icp_align and pred_vertices is not None and len(pred_vertices) > 0 and gt_vertices is not None:
        pred_vertices, _, _ = icp_align(pred_vertices, gt_vertices)

    frames = []

    for i in range(n_frames):
        angle = 360 * i / n_frames

        fig = plt.figure(figsize=(14, 6), facecolor='black')
        if sample_name:
            fig.suptitle(sample_name, fontsize=14, fontweight='bold', color='white')

        # Ground Truth (left)
        ax_gt = fig.add_subplot(121, projection='3d', facecolor='black')
        if gt_vertices is not None and len(gt_vertices) > 0:
            if render_mesh and gt_is_mesh and gt_faces is not None:
                render_mesh_on_axis(ax_gt, gt_vertices, gt_faces, base_color=gt_color)
            else:
                render_pointcloud_with_shading(ax_gt, gt_vertices, base_color=gt_color, point_size=0.3)
        ax_gt.set_title('Ground Truth', color='white', fontsize=12)
        ax_gt.view_init(elev=20, azim=angle)
        ax_gt.set_xlim([-1.2, 1.2])
        ax_gt.set_ylim([-1.2, 1.2])
        ax_gt.set_zlim([-1.2, 1.2])
        ax_gt.set_axis_off()
        ax_gt.set_box_aspect([1, 1, 1])

        # Predicted (right)
        ax_pred = fig.add_subplot(122, projection='3d', facecolor='black')

        if pred_vertices is not None and len(pred_vertices) > 0:
            if render_mesh and pred_is_mesh and pred_faces is not None:
                render_mesh_on_axis(ax_pred, pred_vertices, pred_faces, base_color=pred_color)
            else:
                render_pointcloud_with_shading(ax_pred, pred_vertices, base_color=pred_color, point_size=0.3)
        else:
            ax_pred.text(0, 0, 0, 'Generation\nFailed', ha='center', va='center',
                        fontsize=12, color='red')

        title_suffix = ' (ICP Aligned)' if use_icp_align else ''
        ax_pred.set_title(f'Predicted{title_suffix}', color='white', fontsize=12)
        ax_pred.view_init(elev=20, azim=angle)
        ax_pred.set_xlim([-1.2, 1.2])
        ax_pred.set_ylim([-1.2, 1.2])
        ax_pred.set_zlim([-1.2, 1.2])
        ax_pred.set_axis_off()
        ax_pred.set_box_aspect([1, 1, 1])

        plt.tight_layout()

        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf)[:, :, :3]
        frames.append(image.copy())
        plt.close(fig)

    imageio.mimsave(str(output_path), frames, duration=duration, loop=0)
    return str(output_path)


def plot_slice_with_mask(
    slice_tensor,
    mask_tensor,
    output_path: Optional[Union[str, Path]] = None,
    rotate: bool = True
):
    """
    Plot CT slice with mask overlay.

    Args:
        slice_tensor: (1, H, W) tensor with CT values
        mask_tensor: (1, H, W) binary mask tensor
        output_path: Optional path to save the figure
        rotate: Whether to rotate 90 degrees CCW
    """
    slice_display = slice_tensor.squeeze().numpy()
    mask_display = mask_tensor.squeeze().numpy()

    if rotate:
        slice_display = np.rot90(slice_display)
        mask_display = np.rot90(mask_display)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(slice_display, cmap='gray')
    axes[0].set_title('CT Slice')
    axes[0].axis('off')

    axes[1].imshow(mask_display, cmap='gray')
    axes[1].set_title('Binary Mask')
    axes[1].axis('off')

    axes[2].imshow(slice_display, cmap='gray')
    axes[2].imshow(mask_display, cmap='Reds', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        plt.close(fig)
    else:
        plt.show()
