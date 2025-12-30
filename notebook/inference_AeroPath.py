import numpy as np
import os
from nilearn import image
import nibabel as nib

import torch
import matplotlib.pyplot as plt
from PIL import Image
import requests
from transformers import pipeline

import os
import imageio
import uuid
from inference import Inference, ready_gaussian_for_video_rendering, render_video, load_image, load_single_mask, display_image, make_scene, interactive_visualizer

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

def extract_all_objects_middle_slices(intensity_nii_path, mask_nii_path, axis=2):
    """
    Function 2: Iteratively obtains the segmented object from the middle slice 
    of each unique label found in the mask.
    Returns a Dictionary: { label_id: 2D_numpy_array }
    """
    # Load data
    img_nii = nib.load(intensity_nii_path)
    mask_nii = nib.load(mask_nii_path)
    
    img_data = img_nii.get_fdata()
    mask_data = mask_nii.get_fdata()
    
    # Handle 4D images (take first volume)
    if img_data.ndim == 4:
        img_data = img_data[..., 0]

    # Find all unique objects (excluding background 0)
    unique_labels = np.unique(mask_data)
    object_ids = unique_labels[unique_labels != 0].astype(int)
    
    results = {}
    
    print(f"Found {len(object_ids)} objects. Extracting middle slices...")
    
    for label_id in object_ids:
        # 1. Determine the middle slice for this specific object
        mid_idx = get_object_middle_index(mask_data, label_id, axis)
        
        if mid_idx is None:
            continue
            
        # 2. Slice the arrays
        if axis == 0:
            slice_img = img_data[mid_idx, :, :]
            slice_mask = mask_data[mid_idx, :, :]
        elif axis == 1:
            slice_img = img_data[:, mid_idx, :]
            slice_mask = mask_data[:, mid_idx, :]
        else: # axis == 2
            slice_img = img_data[:, :, mid_idx]
            slice_mask = mask_data[:, :, mid_idx]
            
        # 3. Apply mask (Segment the object)
        # Only keep pixels belonging to THIS label
        segmented_slice = np.where(slice_mask == label_id, slice_img, 0)
        
        # 4. Optional: Crop 2D to remove excess black background
        # This makes visualization much better
        non_zero = np.argwhere(segmented_slice)
        if non_zero.size > 0:
            top_left = non_zero.min(axis=0)
            bottom_right = non_zero.max(axis=0)
            segmented_slice = segmented_slice[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]
            
        results[label_id] = segmented_slice
        
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

config_path = f"/PHShome/yl535/project/python/sam_3d/sam-3d-objects/checkpoints/checkpoints/pipeline.yaml"
inference = Inference(config_path, compile=False)

device = 0 if torch.cuda.is_available() else -1
generator = pipeline(
    "mask-generation", 
    model="facebook/sam2-hiera-large", 
    device=device,
    torch_dtype=torch.float32 
)

type = "lungs"  # "airways" or "lungs"
input_folder = "/PHShome/yl535/project/python/datasets/AeroPath/data" 
output_dir = f"/PHShome/yl535/project/python/sam_3d/sam-3d-objects/results_AeroPath/" 

pairs = get_image_mask_pairs(input_folder, type=type)

print(f"\nSuccessfully loaded {len(pairs)} pairs.")
os.makedirs(output_dir, exist_ok=True)
# Example loop to process them
for img_path, mask_path in pairs:
    filename = os.path.basename(img_path)
    base_name = filename.split('.nii')[0] 
    print(f"Processing: {os.path.basename(img_path)}, {os.path.basename(mask_path)}")

    extracted_objects = extract_all_objects_middle_slices(img_path, mask_path, axis=2)
    for label_id, slice_data in extracted_objects.items():
        print(label_id)
        print(type(slice_data))
        save_path = os.path.join(output_dir, f"{base_name}_{label_id:02d}.nii.gz")
        nib.save(slice_data, save_path)

    seg_slice = get_middle_slice_segmentation(img_path, mask_path, axis=2)

    output = inference(image_np, masks[mask_index], seed=42)

    # export gaussian splat (as point cloud)
    output["gs"].save_ply(f"{PATH}/gaussians/single/{IMAGE_NAME}.ply")

    # segment_and_crop_objects(img_path, mask_path, output_dir=output_dir)

    # output = inference(image_np, masks[mask_index], seed=42)
    # output["gs"].save_ply(f"{PATH}/gaussians/single/{IMAGE_NAME}.ply")