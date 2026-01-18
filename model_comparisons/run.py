"""
Run Hunyuan3D-2.1 3D reconstruction on medical imaging datasets.

Processes each sample, generates 3D mesh, saves point clouds, and computes evaluation metrics.
Supports multiple datasets: AeroPath, BCTV, MSD.

Outputs:
    outputs/{dataset_name}/{sample_name}/
        input.png - 2D input image
        mesh.glb - Generated mesh file
        gt_mesh.glb - Ground truth mesh (from point cloud via marching cubes)
        pred_mesh.glb - Predicted mesh (ICP aligned)
    outputs/{dataset_name}/metrics.csv - Metrics for all samples
"""

import sys
import os
import time
import csv
import argparse
from pathlib import Path

import yaml

# Add paths for hy3dgen
sys.path.insert(0, 'models/Hunyuan3D-2')

import numpy as np
import torch
import trimesh

from dataloaders import AeroPathDataset, BCTVAbdomenDataset, MSDDataset
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

# Import from organized utils modules
from utils.preprocessing import prepare_slice_for_hunyuan, normalize_to_unit_range
from utils.metrics import evaluate_reconstruction, icp_align
from utils.tools import mesh_to_pointcloud, normalize_points, save_mesh


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_sample_name(dataset, idx):
    """Get the sample name including file ID and category (organ/mask type)."""
    if isinstance(dataset, AeroPathDataset):
        patient_dir, _, mask_type = dataset.samples[idx]
        patient_id = patient_dir.name
        return f"{patient_id}_{mask_type}"
    elif isinstance(dataset, BCTVAbdomenDataset):
        image_path, _, organ_label = dataset.samples[idx]
        img_id = image_path.stem.replace("img", "")
        organ_name = dataset.ORGAN_LABELS.get(organ_label, f"label{organ_label}")
        # Sanitize organ name for filesystem (replace spaces)
        organ_name = organ_name.replace(" ", "_")
        return f"img{img_id}_{organ_name}"
    elif isinstance(dataset, MSDDataset):
        image_path, _, class_label = dataset.samples[idx]
        # Get directory name (e.g., "lung_001.nii")
        dir_name = image_path.parent.name.replace(".nii", "")
        return f"{dir_name}_class{class_label}"
    else:
        return f"sample_{idx}"


def process_sample(idx, dataset, pipeline, output_base_dir, num_inference_steps):
    """
    Process a single sample: generate mesh, save visualizations, compute metrics.
    Uses ICP alignment for visualization and metrics.
    """
    sample_name = get_sample_name(dataset, idx)
    print(f"\n{'='*60}")
    print(f"Processing sample {idx + 1}/{len(dataset)}: {sample_name}")
    print(f"{'='*60}")

    # Create output directory
    output_dir = output_base_dir / sample_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data (4th return value is mask_type, unused here)
    slice_2d, mask_2d, point_cloud, _ = dataset[idx]
    gt_points = point_cloud.numpy()

    print(f"  Slice shape: {slice_2d.shape}")
    print(f"  Point cloud shape: {gt_points.shape}")

    # Prepare input image (with rotation for correct orientation)
    input_image = prepare_slice_for_hunyuan(
        slice_2d, mask_2d,
        use_mask_overlay=True,
        rotate=True
    )
    input_image.save(output_dir / "input.png")
    print(f"  Saved input image")

    # Generate 3D mesh
    print(f"  Generating 3D mesh...")
    start_time = time.time()

    mesh = None
    try:
        mesh = pipeline(image=input_image, num_inference_steps=num_inference_steps)[0]
        gen_time = time.time() - start_time
        print(f"  Generation time: {gen_time:.2f}s")

        # Save mesh
        if mesh is not None:
            save_mesh(mesh, output_dir / "mesh.glb")
            print(f"  Saved mesh")
    except Exception as e:
        print(f"  ERROR during mesh generation: {e}")
        mesh = None
        gen_time = -1

    # Initialize metrics
    metrics = {
        'file_name': sample_name,
        'generation_time': gen_time,
        'fscore': np.nan,
        'voxel_iou': np.nan,
        'chamfer_distance': np.nan,
        'emd': np.nan,
        'precision': np.nan,
        'recall': np.nan,
        'voxel_dice': np.nan,
    }

    # Normalize GT to [-1, 1] range
    gt_points_norm = normalize_to_unit_range(gt_points)

    if mesh is not None:
        # Get mesh vertices and normalize
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()

        pred_vertices = np.array(mesh.vertices)
        pred_points_norm = normalize_to_unit_range(pred_vertices)

        print(f"  GT points: {len(gt_points_norm)}, Pred vertices: {len(pred_points_norm)}")

        # Compute metrics with ICP alignment
        print(f"  Computing metrics (with ICP alignment)...")
        eval_metrics = evaluate_reconstruction(
            pc_pred=pred_points_norm,
            pc_gt=gt_points_norm,
            threshold=0.01,
            grid_size=64,
            normalize=False,  # Already normalized
            align=True  # Use ICP alignment
        )
        metrics.update(eval_metrics)

        print(f"  F-score: {metrics['fscore']:.4f}, CD: {metrics['chamfer_distance']:.4f}, IoU: {metrics['voxel_iou']:.4f}")

        # Save GT as mesh (voxelized for shading)
        gt_mesh = trimesh.voxel.ops.points_to_marching_cubes(gt_points_norm, pitch=0.02)
        gt_mesh.export(output_dir / "gt_mesh.glb")
        print(f"  Saved GT mesh")

        # ICP align predicted mesh to GT and save
        _, transformation, _ = icp_align(pred_points_norm, gt_points_norm)
        pred_mesh_aligned = mesh.copy()
        # Normalize vertices then apply ICP transformation
        pred_verts_norm = normalize_to_unit_range(np.array(pred_mesh_aligned.vertices))
        pred_mesh_aligned.vertices = pred_verts_norm
        pred_mesh_aligned.apply_transform(transformation)
        pred_mesh_aligned.export(output_dir / "pred_mesh.glb")
        print(f"  Saved predicted mesh (ICP aligned)")
    else:
        print(f"ERROR: Mesh generation failed")
    
    print(f"  Done with {sample_name}")
    return metrics


def get_dataset(dataset_name, root_dir, plane):
    """Factory function to create the appropriate dataset based on name."""
    dataset_map = {
        "aeropath": AeroPathDataset,
        "bctv": BCTVAbdomenDataset,
        "msd": MSDDataset,
    }

    key = dataset_name.lower()
    if key not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(dataset_map.keys())}")

    dataset_class = dataset_map[key]

    # Different datasets have different constructor arguments
    if key == "aeropath":
        return dataset_class(root_dir=root_dir, plane=plane, surface_only=True)
    else:
        return dataset_class(root_dir=root_dir, plane=plane, split="train")


def run_dataset(config, pipeline):
    """Run reconstruction on a single dataset configuration."""
    dataset_name = config['dataset_name']
    root_dir = config['root_dir']
    plane = config['plane']
    num_inference_steps = config['num_inference_steps']

    print("\n" + "="*60)
    print(f"Processing {dataset_name} Dataset")
    print("="*60)

    output_base_dir = Path("outputs") / dataset_name
    output_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading {dataset_name} dataset...")
    dataset = get_dataset(dataset_name, root_dir, plane)
    print(f"Total samples: {len(dataset)}")

    all_metrics = []

    for idx in range(len(dataset)):
        try:
            metrics = process_sample(idx, dataset, pipeline, output_base_dir, num_inference_steps)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"ERROR processing sample {idx}: {e}")
            sample_name = get_sample_name(dataset, idx)
            all_metrics.append({
                'file_name': sample_name,
                'generation_time': -1,
                'fscore': np.nan,
                'voxel_iou': np.nan,
                'chamfer_distance': np.nan,
                'emd': np.nan,
                'precision': np.nan,
                'recall': np.nan,
                'voxel_dice': np.nan,
            })

        break

    # Save metrics to CSV
    csv_path = output_base_dir / "metrics.csv"
    print(f"\nSaving metrics to {csv_path}")

    fieldnames = ['file_name', 'generation_time', 'fscore', 'voxel_iou',
                  'chamfer_distance', 'emd', 'precision', 'recall', 'voxel_dice']

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Summary Statistics for {dataset_name}")
    print(f"{'='*60}")

    valid_metrics = [m for m in all_metrics if not np.isnan(m['fscore'])]
    print(f"Successful: {len(valid_metrics)}/{len(all_metrics)}")

    if valid_metrics:
        for name in ['fscore', 'chamfer_distance', 'voxel_iou', 'emd', 'precision', 'recall', 'voxel_dice']:
            values = [m[name] for m in valid_metrics]
            print(f"  {name}: {np.mean(values):.4f} +/- {np.std(values):.4f}")

    print(f"\nResults saved to {output_base_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run Hunyuan3D-2.1 3D reconstruction")
    parser.add_argument("--config", type=str, nargs='+', required=True,
                        help="Path(s) to config YAML file(s)")
    args = parser.parse_args()

    print("="*60)
    print("Hunyuan3D-2.1 3D Reconstruction")
    print("="*60)

    os.environ['HY3DGEN_MODELS'] = 'models'

    # Load model once
    print(f"\nLoading Hunyuan3D-2.1 model...")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        'Hunyuan3D-2.1',
        subfolder='hunyuan3d-dit-v2-1',
        use_safetensors=False
    )
    print("Model loaded successfully")

    # Process each config file
    for config_path in args.config:
        print(f"\n{'#'*60}")
        print(f"Loading config: {config_path}")
        print(f"{'#'*60}")

        config = load_config(config_path)
        run_dataset(config, pipeline)

    print("\n" + "="*60)
    print("All datasets processed!")
    print("="*60)


if __name__ == "__main__":
    main()
