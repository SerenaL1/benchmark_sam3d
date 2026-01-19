import numpy as np
import os
import nibabel as nib

import torch
import matplotlib.pyplot as plt
from PIL import Image
import argparse

import imageio
from pathlib import Path
from typing import List, Tuple

from inference import (
    Inference, 
    ready_gaussian_for_video_rendering, 
    render_video, 
    make_scene
)

def get_object_middle_index(mask_data, label_id, axis=2):
    """
    Determines the middle slice index for a SPECIFIC object label.
    """
    object_indices = np.where(mask_data == label_id)
    
    if len(object_indices[0]) == 0:
        return None
        
    axis_indices = object_indices[axis]
    min_idx = np.min(axis_indices)
    max_idx = np.max(axis_indices)
    middle_index = (min_idx + max_idx) // 2
    
    return middle_index

def to_uint8_image(arr):
    """
    Normalize array to [0, 255] uint8
    """
    if arr.max() == arr.min():
        arr_normalized = np.zeros_like(arr)
    else:
        arr_normalized = (arr - arr.min()) / (arr.max() - arr.min())
    
    arr_scaled = arr_normalized * 255
    return arr_scaled.astype(np.uint8)

def extract_all_objects_middle_slices(intensity_nii_path, mask_nii_path, axis=2):
    """
    Extract middle slice for each tumor region in the mask.
    Returns: List of (slice_img, slice_mask, label_id)
    """
    # Load data
    img_nii = nib.load(intensity_nii_path)
    mask_nii = nib.load(mask_nii_path)
    
    img_data = img_nii.get_fdata()
    mask_data = mask_nii.get_fdata()
    
    # Handle 4D data
    if img_data.ndim == 4:
        img_data = img_data[..., 0]
    if mask_data.ndim == 4:
        mask_data = mask_data[..., 0]

    # Find unique tumor labels (exclude background 0)
    unique_labels = np.unique(mask_data)
    object_ids = unique_labels[unique_labels != 0].astype(int)
    
    print(f"Found {len(object_ids)} tumor regions: {object_ids}")
    
    results = []
    for label_id in object_ids:
        mid_idx = get_object_middle_index(mask_data, label_id, axis)
        
        if mid_idx is None:
            print(f"  Warning: Could not find middle index for label {label_id}")
            continue
            
        # Extract slice based on axis
        if axis == 0:
            slice_img = img_data[mid_idx, :, :]
            slice_mask = mask_data[mid_idx, :, :]
        elif axis == 1:
            slice_img = img_data[:, mid_idx, :]
            slice_mask = mask_data[:, mid_idx, :]
        else: # axis == 2 (axial - standard for brain)
            slice_img = img_data[:, :, mid_idx]
            slice_mask = mask_data[:, :, mid_idx]
        
        # Convert to RGB uint8 (SAM3D expects this format)
        slice_img = np.repeat(slice_img[:, :, np.newaxis], 3, axis=2)
        slice_img = to_uint8_image(slice_img)
        slice_mask = np.squeeze(slice_mask)
        
        print(f"  Extracted label {label_id}: slice {mid_idx}, shape {slice_img.shape}")
        results.append((slice_img, slice_mask, label_id))
            
    return results

def find_image_mask_pairs(input_folder: str, recursive: bool = True) -> List[Tuple[Path, Path]]:
    """
    Find matching image-mask pairs in the input folder.
    Expects files like: filename.nii.gz and filename_mask.nii.gz
    """
    p = Path(input_folder)
    it = p.rglob("*.nii.gz") if recursive else p.glob("*.nii.gz")

    images = {}  # key -> image path
    masks  = {}  # key -> mask path

    for f in it:
        name = f.name
        if name.endswith("_mask.nii.gz"):
            # e.g., "TCGA-CS-5396_2001.03.02_flair_1_mask.nii.gz"
            key = name[:-len("_mask.nii.gz")]
            masks[key] = f
        elif name.endswith(".nii.gz"):
            # e.g., "TCGA-CS-5396_2001.03.02_flair_1.nii.gz"
            key = name[:-len(".nii.gz")]
            images[key] = f

    # Match images with their masks
    pairs = [(images[k], masks[k]) for k in sorted(images.keys() & masks.keys())]
    
    print(f"Found {len(pairs)} image-mask pairs in {input_folder}")
    return pairs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM3D Inference for BraTS Dataset")
    
    parser.add_argument(
        '--config_path', 
        type=str, 
        required=True,
        help='Path to the SAM3D pipeline.yaml configuration file'
    )
    parser.add_argument(
        '--input_folder', 
        type=str, 
        required=True,
        help='Path to preprocessed BraTS segmented folder'
    )
    parser.add_argument(
        '--output_folder', 
        type=str, 
        required=True,
        help='Path to save SAM3D predictions and visualizations'
    )
    parser.add_argument(
        '--slice_axis', 
        type=int, 
        default=2,
        help='Axis for slicing (0=sagittal, 1=coronal, 2=axial). Default: 2 (axial)'
    )
    
    args = parser.parse_args()

    print("="*60)
    print("SAM3D Inference for BraTS")
    print("="*60)
    print(f"Config: {args.config_path}")
    print(f"Input:  {args.input_folder}")
    print(f"Output: {args.output_folder}")
    print(f"Axis:   {args.slice_axis}")
    print("="*60)

    # Initialize SAM3D inference
    print("\nInitializing SAM3D model...")
    inference = Inference(args.config_path, compile=False)

    # Create output directory
    os.makedirs(args.output_folder, exist_ok=True)

    # Find all image-mask pairs
    print("\nFinding image-mask pairs...")
    pairs = find_image_mask_pairs(args.input_folder)

    if len(pairs) == 0:
        print("ERROR: No image-mask pairs found!")
        print("Check that your input folder contains files like:")
        print("  - filename.nii.gz")
        print("  - filename_mask.nii.gz")
        exit(1)

    # Process each pair
    print(f"\nProcessing {len(pairs)} pairs...\n")
    
    for idx, (scan_path, mask_path) in enumerate(pairs):
        print(f"[{idx+1}/{len(pairs)}] Processing:")
        print(f"  Image: {scan_path.name}")
        print(f"  Mask:  {mask_path.name}")
        
        try:
            # Extract middle slices for each tumor region
            objs = extract_all_objects_middle_slices(
                scan_path, 
                mask_path, 
                axis=args.slice_axis
            )
            
            if len(objs) == 0:
                print(f"  Warning: No objects found in {scan_path.name}, skipping...")
                continue
            
            # Process first tumor region (you can loop through all if needed)
            extracted_scan, extracted_mask, label_id = objs[0]
            
            print(f"  Running SAM3D inference on label {label_id}...")
            output = inference(extracted_scan, extracted_mask, seed=42)

            # Save 3D reconstruction as PLY
            output_name = f"{scan_path.stem}_label{label_id}"
            ply_path = os.path.join(args.output_folder, f"{output_name}.ply")
            output["gs"].save_ply(ply_path)
            print(f"  Saved: {ply_path}")

            # Generate and save visualization video
            print(f"  Generating visualization...")
            scene_gs = make_scene(output)
            scene_gs = ready_gaussian_for_video_rendering(scene_gs)

            video = render_video(
                scene_gs,
                r=1,
                fov=60,
                pitch_deg=15,
                yaw_start_deg=-45,
                resolution=512,
            )["color"]

            gif_path = os.path.join(args.output_folder, f"{output_name}.gif")
            imageio.mimsave(
                gif_path,
                video,
                format="GIF",
                duration=1000 / 30,
                loop=0,
            )
            print(f"  Saved: {gif_path}")
            
        except Exception as e:
            print(f"  ERROR processing {scan_path.name}: {e}")
            continue
        
        print()

    print("="*60)
    print("✅ Processing complete!")
    print(f"Results saved to: {args.output_folder}")
    print("="*60)
