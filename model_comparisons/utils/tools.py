"""
Utility tools for mesh and point cloud operations.

Functions for loading, saving, and manipulating 3D geometry.
"""

import numpy as np
import trimesh
from pathlib import Path
from typing import Union, Optional, Tuple


def load_mesh(path: Union[str, Path]) -> trimesh.Trimesh:
    """
    Load a mesh from file.

    Args:
        path: Path to mesh file (supports OBJ, PLY, GLB, etc.)

    Returns:
        Trimesh object
    """
    mesh = trimesh.load(str(path))
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    return mesh


def save_mesh(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    path: Union[str, Path],
    file_type: Optional[str] = None
) -> str:
    """
    Save a mesh to file.

    Args:
        mesh: Trimesh mesh or Scene
        path: Output path
        file_type: Optional file type override

    Returns:
        Path to saved file
    """
    path = Path(path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()

    mesh.export(str(path), file_type=file_type)
    return str(path)


def mesh_to_pointcloud(
    mesh: Union[trimesh.Trimesh, trimesh.Scene],
    n_points: int = 10000
) -> np.ndarray:
    """
    Sample points from mesh surface.

    Args:
        mesh: Trimesh mesh or Scene
        n_points: Number of points to sample

    Returns:
        Point cloud array (N, 3)
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()

    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points


def normalize_points(
    points: np.ndarray,
    center: Optional[np.ndarray] = None,
    scale: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normalize point cloud to [-1, 1] range.

    Args:
        points: Point cloud (N, 3)
        center: Optional pre-computed center
        scale: Optional pre-computed scale

    Returns:
        Tuple of (normalized_points, center, scale)
    """
    if center is None:
        center = (points.max(axis=0) + points.min(axis=0)) / 2
    if scale is None:
        scale = (points.max(axis=0) - points.min(axis=0)).max() / 2
        if scale == 0:
            scale = 1.0

    normalized = (points - center) / scale
    return normalized, center, scale


def get_mesh_vertices(
    mesh: Union[trimesh.Trimesh, trimesh.Scene]
) -> np.ndarray:
    """
    Extract vertices from mesh.

    Args:
        mesh: Trimesh mesh or Scene

    Returns:
        Vertices array (N, 3)
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    return np.array(mesh.vertices)


def get_mesh_faces(
    mesh: Union[trimesh.Trimesh, trimesh.Scene]
) -> np.ndarray:
    """
    Extract faces from mesh.

    Args:
        mesh: Trimesh mesh or Scene

    Returns:
        Faces array (F, 3)
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_geometry()
    return np.array(mesh.faces)


def compute_mesh_bounds(
    mesh: Union[trimesh.Trimesh, trimesh.Scene]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mesh bounding box.

    Args:
        mesh: Trimesh mesh or Scene

    Returns:
        Tuple of (min_bounds, max_bounds)
    """
    vertices = get_mesh_vertices(mesh)
    return vertices.min(axis=0), vertices.max(axis=0)


def subsample_points(
    points: np.ndarray,
    n_points: int,
    method: str = 'random'
) -> np.ndarray:
    """
    Subsample a point cloud.

    Args:
        points: Point cloud (N, 3)
        n_points: Target number of points
        method: 'random' or 'farthest' (farthest point sampling)

    Returns:
        Subsampled point cloud
    """
    if len(points) <= n_points:
        return points

    if method == 'random':
        idx = np.random.choice(len(points), n_points, replace=False)
        return points[idx]
    elif method == 'farthest':
        # Farthest point sampling
        selected = [np.random.randint(len(points))]
        distances = np.full(len(points), np.inf)

        for _ in range(n_points - 1):
            last_point = points[selected[-1]]
            new_distances = np.linalg.norm(points - last_point, axis=1)
            distances = np.minimum(distances, new_distances)
            selected.append(np.argmax(distances))

        return points[selected]
    else:
        raise ValueError(f"Unknown subsampling method: {method}")
