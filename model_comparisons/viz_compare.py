"""
Visualize comparison between GT and reconstructed point clouds.

Loads saved point clouds from run.py outputs and generates comparison visualizations
(side-by-side rotating GIF with ICP alignment).

Usage:
    python viz_compare.py                          # Process all samples
    python viz_compare.py --sample AEROPATH_001    # Process specific sample
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio


# Configuration
DATASET_NAME = "AeroPath"
OUTPUT_BASE_DIR = Path("outputs") / DATASET_NAME
N_GIF_FRAMES = 36
GIF_DURATION = 0.1


def points_to_mesh(points: np.ndarray):
    """Convert point cloud to mesh using convex hull."""
    try:
        cloud = trimesh.PointCloud(points)
        return cloud.convex_hull
    except Exception:
        return None


def render_mesh_on_axis(ax, vertices, faces, base_color=(0.7, 0.7, 0.7)):
    """Render a mesh with lighting on a matplotlib 3D axis."""
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
    intensities = 0.3 + 0.7 * intensities

    # Apply shading colors
    colors = np.zeros((len(faces), 4))
    colors[:, 0] = intensities * base_color[0]
    colors[:, 1] = intensities * base_color[1]
    colors[:, 2] = intensities * base_color[2]
    colors[:, 3] = 1.0
    mesh_collection.set_facecolors(colors)

    ax.add_collection3d(mesh_collection)


def create_comparison_gif(
    gt_points: np.ndarray,
    pred_points: np.ndarray,
    output_path: Path,
    sample_name: str = "",
    n_frames: int = 36,
    duration: float = 0.1,
    render_mesh: bool = True,
    gt_color: tuple = (0.29, 0.56, 0.85),
    pred_color: tuple = (0.7, 0.7, 0.7)
):
    """
    Create a side-by-side rotating GIF comparing GT and predicted point clouds.

    Args:
        gt_points: Ground truth point cloud (N, 3)
        pred_points: Predicted point cloud (M, 3), already ICP aligned
        output_path: Path to save the GIF
        sample_name: Name to display as title
        n_frames: Number of frames
        duration: Duration per frame
        render_mesh: If True, render as solid mesh; if False, render as point cloud
        gt_color: RGB color for GT
        pred_color: RGB color for predicted
    """
    # Convert to meshes if requested
    gt_mesh = points_to_mesh(gt_points) if render_mesh else None
    pred_mesh = points_to_mesh(pred_points) if render_mesh else None

    gt_vertices = np.array(gt_mesh.vertices) if gt_mesh else gt_points
    gt_faces = np.array(gt_mesh.faces) if gt_mesh else None

    pred_vertices = np.array(pred_mesh.vertices) if pred_mesh else pred_points
    pred_faces = np.array(pred_mesh.faces) if pred_mesh else None

    frames = []

    for i in range(n_frames):
        angle = 360 * i / n_frames

        fig = plt.figure(figsize=(14, 6), facecolor='black')
        if sample_name:
            fig.suptitle(sample_name, fontsize=14, fontweight='bold', color='white')

        # Ground Truth (left)
        ax_gt = fig.add_subplot(121, projection='3d', facecolor='black')
        if gt_faces is not None:
            render_mesh_on_axis(ax_gt, gt_vertices, gt_faces, base_color=gt_color)
        else:
            ax_gt.scatter(
                gt_points[:, 0], gt_points[:, 1], gt_points[:, 2],
                c='#4A90D9', s=0.3, alpha=0.8
            )
        ax_gt.set_title('Ground Truth', color='white', fontsize=12)
        ax_gt.view_init(elev=20, azim=angle)
        ax_gt.set_xlim([-1.2, 1.2])
        ax_gt.set_ylim([-1.2, 1.2])
        ax_gt.set_zlim([-1.2, 1.2])
        ax_gt.set_axis_off()
        ax_gt.set_box_aspect([1, 1, 1])

        # Predicted (right)
        ax_pred = fig.add_subplot(122, projection='3d', facecolor='black')
        if pred_faces is not None:
            render_mesh_on_axis(ax_pred, pred_vertices, pred_faces, base_color=pred_color)
        else:
            ax_pred.scatter(
                pred_points[:, 0], pred_points[:, 1], pred_points[:, 2],
                c='#D94A4A', s=0.3, alpha=0.8
            )
        ax_pred.set_title('Predicted (ICP Aligned)', color='white', fontsize=12)
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
    print(f"  Saved comparison GIF: {output_path}")


def process_sample(sample_dir: Path):
    """Process a single sample directory and create comparison GIF."""
    gt_path = sample_dir / "gt_pointcloud.ply"
    pred_path = sample_dir / "pred_pointcloud.ply"

    if not gt_path.exists():
        print(f"  Skipping {sample_dir.name}: gt_pointcloud.ply not found")
        return False

    if not pred_path.exists():
        print(f"  Skipping {sample_dir.name}: pred_pointcloud.ply not found")
        return False

    # Load point clouds
    gt_pcd = trimesh.load(gt_path)
    pred_pcd = trimesh.load(pred_path)

    gt_points = np.array(gt_pcd.vertices)
    pred_points = np.array(pred_pcd.vertices)

    print(f"  GT points: {len(gt_points)}, Pred points: {len(pred_points)}")

    # Create comparison GIF
    output_path = sample_dir / "comparison.gif"
    create_comparison_gif(
        gt_points=gt_points,
        pred_points=pred_points,
        output_path=output_path,
        sample_name=sample_dir.name,
        n_frames=N_GIF_FRAMES,
        duration=GIF_DURATION,
        render_mesh=True
    )

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparison visualizations from saved point clouds"
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="Process specific sample (e.g., AEROPATH_001_CT_HR)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory (default: {OUTPUT_BASE_DIR})"
    )
    parser.add_argument(
        "--no-mesh",
        action="store_true",
        help="Render as point cloud instead of mesh"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_BASE_DIR

    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return

    if args.sample:
        # Process specific sample
        sample_dir = output_dir / args.sample
        if not sample_dir.exists():
            print(f"Sample directory not found: {sample_dir}")
            return
        print(f"Processing {args.sample}...")
        process_sample(sample_dir)
    else:
        # Process all samples
        sample_dirs = [d for d in output_dir.iterdir() if d.is_dir()]
        print(f"Found {len(sample_dirs)} sample directories")

        success_count = 0
        for sample_dir in sorted(sample_dirs):
            print(f"\nProcessing {sample_dir.name}...")
            if process_sample(sample_dir):
                success_count += 1

        print(f"\n{'='*60}")
        print(f"Processed {success_count}/{len(sample_dirs)} samples successfully")


if __name__ == "__main__":
    main()
