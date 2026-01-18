import numpy as np
import os
from nilearn import image
import nibabel as nib

import torch
import matplotlib.pyplot as plt
from PIL import Image
import requests
from transformers import pipeline
from pathlib import Path
from typing import List, Tuple
import argparse

import os
import imageio
import uuid
from inference import Inference, ready_gaussian_for_video_rendering, render_video, load_image, load_single_mask, display_image, make_scene, interactive_visualizer

def get_image_mask_pairs(root_dir, type="airways"):
    """
    Traverses subfolders named [id] and finds matching CT/Mask pairs.
    """
    data_pairs = []
    
    # 1. Get all subfolders in the root directory
    # We filter to ensure we only look at directories
    subfolders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    # 2. Sort them numerically (so we process 1, 2, ... 10, instead of 1, 10, 2)
    # If folder names are not pure integers, this falls back to string sorting
    try:
        subfolders.sort(key=lambda x: int(x))
    except ValueError:
        subfolders.sort()

    print(f"Found {len(subfolders)} patient folders.")

    for case_id in subfolders:
        case_path = os.path.join(root_dir, case_id)
        
        # 3. Construct specific filenames based on the pattern
        # Pattern: [id]_CT_HR.nii.gz AND [id]_CT_HR_label_airways.nii.gz
        img_name = f"{case_id}_CT_HR.nii.gz"
        mask_name = f"{case_id}_CT_HR_label_{type}.nii.gz"
        
        img_full_path = os.path.join(case_path, img_name)
        mask_full_path = os.path.join(case_path, mask_name)
        
        # 4. Verify files exist before adding to list
        if os.path.exists(img_full_path) and os.path.exists(mask_full_path):
            data_pairs.append((img_full_path, mask_full_path))
        else:
            print(f"Warning: Missing files in folder {case_id}")
            # Optional: Check which one is missing for debugging
            if not os.path.exists(img_full_path): print(f"  - Missing: {img_name}")
            if not os.path.exists(mask_full_path): print(f"  - Missing: {mask_name}")

    return data_pairs

def show_masks(image, masks):
    """
    Helper function to display the image with masks overlaid.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    # Iterate through each mask and overlay it
    for mask in masks:
        # Convert PIL mask to numpy array (boolean)
        m = np.array(mask) > 0 
        
        # Generate a random color for this mask
        color = np.concatenate([np.random.random(3), [0.6]]) # [R, G, B, Alpha]
        
        # Create a colored mask image
        h, w = m.shape
        mask_image = m.reshape(h, w, 1) * color.reshape(1, 1, -1)
        
        # Overlay the mask
        ax.imshow(mask_image)

    plt.axis('off')
    plt.show()

def get_object_middle_index(mask_data, label_id, axis=2):
    """
    Function 1: Determines the middle slice index for a SPECIFIC object label.
    It finds the bounding box of the label and calculates the center.
    """
    # Find the indices where the mask equals the current label
    # np.where returns a tuple of arrays (z, y, x)
    object_indices = np.where(mask_data == label_id)
    
    # Check if object exists
    if len(object_indices[0]) == 0:
        return None
        
    # Get indices for the specific axis
    axis_indices = object_indices[axis]
    
    # Calculate center: (min + max) // 2
    min_idx = np.min(axis_indices)
    max_idx = np.max(axis_indices)
    
    middle_index = (min_idx + max_idx) // 2
    
    return middle_index

# def extract_all_objects_middle_slices(intensity_nii_path, mask_nii_path, axis=2):
#     """
#     Function 2: Iteratively obtains the segmented object from the middle slice 
#     of each unique label found in the mask.
#     Returns a Dictionary: { label_id: 2D_numpy_array }
#     """
#     # Load data
#     img_nii = nib.load(intensity_nii_path)
#     mask_nii = nib.load(mask_nii_path)
    
#     img_data = img_nii.get_fdata()
#     mask_data = mask_nii.get_fdata()
    
#     # Handle 4D images (take first volume)
#     if img_data.ndim == 4:
#         img_data = img_data[..., 0]

#     # Find all unique objects (excluding background 0)
#     unique_labels = np.unique(mask_data)
#     object_ids = unique_labels[unique_labels != 0].astype(int)
    
#     results = {}
    
#     print(f"Found {len(object_ids)} objects. Extracting middle slices...")
    
#     for label_id in object_ids:
#         # 1. Determine the middle slice for this specific object
#         mid_idx = get_object_middle_index(mask_data, label_id, axis)
        
#         if mid_idx is None:
#             continue
            
#         # 2. Slice the arrays
#         if axis == 0:
#             slice_img = img_data[mid_idx, :, :]
#             slice_mask = mask_data[mid_idx, :, :]
#         elif axis == 1:
#             slice_img = img_data[:, mid_idx, :]
#             slice_mask = mask_data[:, mid_idx, :]
#         else: # axis == 2
#             slice_img = img_data[:, :, mid_idx]
#             slice_mask = mask_data[:, :, mid_idx]
            
#         # 3. Apply mask (Segment the object)
#         # Only keep pixels belonging to THIS label
#         segmented_slice = np.where(slice_mask == label_id, slice_img, 0)
        
#         # 4. Optional: Crop 2D to remove excess black background
#         # This makes visualization much better
#         non_zero = np.argwhere(segmented_slice)
#         if non_zero.size > 0:
#             top_left = non_zero.min(axis=0)
#             bottom_right = non_zero.max(axis=0)
#             segmented_slice = segmented_slice[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
            
#         results[label_id] = segmented_slice
        
#     return results

def to_uint8_image(arr):
    # 1. Normalize array to range [0, 1]
    # (Avoid division by zero if max == min)
    if arr.max() == arr.min():
        arr_normalized = np.zeros_like(arr)
    else:
        arr_normalized = (arr - arr.min()) / (arr.max() - arr.min())
    
    # 2. Scale to [0, 255]
    arr_scaled = arr_normalized * 255
    
    # 3. Cast to uint8
    return arr_scaled.astype(np.uint8)

def extract_all_objects_middle_slices(intensity_nii_path, mask_nii_path, axis=2):
    # Load data
    img_nii = nib.load(intensity_nii_path)
    mask_nii = nib.load(mask_nii_path)
    
    img_data = img_nii.get_fdata()
    mask_data = mask_nii.get_fdata()
    
    # --- FIX 1: Handle 4D masks (just like you did for images) ---
    if img_data.ndim == 4:
        img_data = img_data[..., 0]
    if mask_data.ndim == 4:
        mask_data = mask_data[..., 0]

    unique_labels = np.unique(mask_data)
    object_ids = unique_labels[unique_labels != 0].astype(int)
    
    print(f"Found {len(object_ids)} objects. Extracting middle slices...")
    
    results = []
    for label_id in object_ids:
        mid_idx = get_object_middle_index(mask_data, label_id, axis)
        
        if mid_idx is None:
            continue
            
        if axis == 0:
            slice_img = img_data[mid_idx, :, :]
            slice_mask = mask_data[mid_idx, :, :]
        elif axis == 1:
            slice_img = img_data[:, mid_idx, :]
            slice_mask = mask_data[:, mid_idx, :]
        else: # axis == 2
            slice_img = img_data[:, :, mid_idx]
            slice_mask = mask_data[:, :, mid_idx]
        
        # 1. Expand Image to 3 Channels (RGB)
        slice_img = np.repeat(slice_img[:, :, np.newaxis], 3, axis=2)
        slice_img = to_uint8_image(slice_img)
        slice_mask = np.squeeze(slice_mask)
        
        results.append((slice_img, slice_mask, label_id))
            
    return results

def visualize_segmented_object(segmented_slice, label_id):
    """
    Function 3: Visualizes the segmented object.
    """
    # Rotate for standard orientation
    to_show = np.rot90(segmented_slice)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(to_show, cmap='gray')
    plt.title(f"Object Label: {label_id}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def find_image_mask_pairs(input_folder: str, recursive: bool = True) -> List[Tuple[Path, Path]]:
    p = Path(input_folder)
    it = p.rglob("*.nii.gz") if recursive else p.glob("*.nii.gz")

    images = {}  # key -> image path
    masks  = {}  # key -> mask path

    for f in it:
        name = f.name
        if name.endswith("_mask.nii.gz"):
            key = name[:-len("_mask.nii.gz")]          # e.g., "10_CT_HR_1"
            masks[key] = f
        else:
            key = name[:-len(".nii.gz")]               # e.g., "10_CT_HR_1"
            images[key] = f

    pairs = [(images[k], masks[k]) for k in sorted(images.keys() & masks.keys())]
    return pairs

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Inference")
    
    parser.add_argument(
        '--config_path', 
        type=str, 
        required=True,
        help='Path to the configuration YAML file.'
    )
    parser.add_argument(
        '--input_folder', 
        type=str, 
        required=True,
        help='Path to the folder containing input NIfTI files.'
    )
    parser.add_argument(
        '--output_folder', 
        type=str, 
        required=True,
        help='Path to the folder to save output results.'
    )
    parser.add_argument(
        '--slice_axis', 
        type=int, 
        default=2,
        help='Path to the folder to save output results.'
    )
    
    args = parser.parse_args()

    config_path = args.config_path # f"/PHShome/yl535/project/python/sam_3d/sam-3d-objects/checkpoints/checkpoints/pipeline.yaml"
    # type = "lungs"  # "airways" or "lungs"
    input_folder = args.input_folder # f"/PHShome/yl535/project/python/datasets/AeroPath/lungs_segmented" 
    output_dir = args.output_folder  # f"/PHShome/yl535/project/python/sam_3d/sam-3d-objects/results_AeroPath/lungs" 

    inference = Inference(config_path, compile=False)

    device = 0 if torch.cuda.is_available() else -1
    generator = pipeline(
        "mask-generation", 
        model="facebook/sam2-hiera-large", 
        device=device,
        torch_dtype=torch.float32 
    )

    # create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    pairs = find_image_mask_pairs(input_folder)
    for scan_path,mask_path  in pairs:
        print("IMG :", scan_path.name)
        print("MASK:", mask_path.name)
        objs = extract_all_objects_middle_slices(scan_path, mask_path, axis=args.slice_axis)
        extracted_scan, extracted_mask = objs[0][0], objs[0][1]
        # print(f"Shape: {extracted_scan.shape}, Type: {extracted_scan.dtype}")
        # print(f"Unique mask values: {np.unique(extracted_scan)}")
        # print(f"Shape: {extracted_mask.shape}, Type: {extracted_mask.dtype}")
        # print(f"Unique mask values: {np.unique(extracted_mask)}")
        # sys.exit()
        output = inference(extracted_scan, extracted_mask, seed=42)

        output["gs"].save_ply(f"{output_dir}/{scan_path.name}.ply")

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

        # save video as gif
        imageio.mimsave(
            os.path.join(f"{output_dir}/{scan_path.name}.gif"),
            video,
            format="GIF",
            duration=1000 / 30,  # default assuming 30fps from the input MP4
            loop=0,  # 0 means loop indefinitely
        )
