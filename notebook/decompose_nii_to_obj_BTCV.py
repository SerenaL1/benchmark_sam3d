import numpy as np
import os
from nilearn import image
import nibabel as nib
import argparse

def segment_and_crop_objects(intensity_nii_path, mask_nii_path, output_dir="segmented_objects"):
    """
    Splits a multi-label NIfTI mask into individual segmented NIfTI files.
    Applies the mask to the intensity image and crops to the object's bounding box.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from {os.path.basename(intensity_nii_path)}...")
    img_vol = image.load_img(intensity_nii_path)
    mask_vol = image.load_img(mask_nii_path)

    # Extract base name (handling .nii or .nii.gz)
    filename = os.path.basename(intensity_nii_path)
    # This splits 'img0001.nii' -> 'img0001'
    base_name = filename.split('.nii')[0] 
    
    # Get data to find unique labels
    mask_data = mask_vol.get_fdata()
    unique_labels = np.unique(mask_data)
    # Filter out background (0)
    object_ids = unique_labels[unique_labels != 0].astype(int)
    
    print(f"Found {len(object_ids)} objects: {object_ids}")
    
    for label_id in object_ids:
        # print(f"Processing Object {label_id}...")
        
        # 1. Create a binary mask for the current object
        binary_mask = image.math_img(
            f"img == {label_id}", 
            img=mask_vol
        )
        
        # 2. Apply mask to original intensity image
        masked_object = image.math_img(
            "img * mask", 
            img=img_vol, 
            mask=binary_mask
        )
        
        # 3. Crop to Bounding Box
        # We crop the mask first to define the true bounding box of the non-zero label
        cropped_mask = image.crop_img(binary_mask)
        
        # We resample the masked object to the cropped_mask geometry.
        cropped_object = image.resample_to_img(masked_object, cropped_mask)
        
        # 4. Save to file
        # Format: img0001_1.nii.gz (originalName_labelID.nii.gz)
        save_path_img = os.path.join(output_dir, f"{base_name}_{label_id}.nii.gz")
        save_path_mask = os.path.join(output_dir, f"{base_name}_{label_id}_mask.nii.gz")
        
        nib.save(cropped_object, save_path_img)
        nib.save(cropped_mask, save_path_mask)
        
        print(f"  Saved: {os.path.basename(save_path_img)} (Shape: {cropped_object.shape})")


def get_image_mask_pairs_btcv(img_dir, label_dir):
    """
    Scans the imagesTr folder and finds the corresponding file in labelsTr.
    Assumes pattern: imgXXXX.nii matches labelXXXX.nii
    """
    data_pairs = []
    
    # Get all files in images directory that end with .nii or .nii.gz
    img_files = [f for f in os.listdir(img_dir) if f.endswith('.nii') or f.endswith('.nii.gz')]
    
    # Sort to ensure processing order
    img_files.sort()

    print(f"Scanning {len(img_files)} files in image directory...")

    for img_filename in img_files:
        # Construct the full path for the image
        img_full_path = os.path.join(img_dir, img_filename)
        
        # Logic to convert imgXXXX.nii -> labelXXXX.nii
        # We replace the prefix "img" with "label"
        if img_filename.startswith("img"):
            label_filename = img_filename.replace("img", "label", 1)
        else:
            # Fallback if naming convention is slightly different, usually just replacing same extension
            print(f"Warning: File {img_filename} does not start with 'img'. Skipping auto-matching.")
            continue
            
        label_full_path = os.path.join(label_dir, label_filename)
        
        # Verify files exist before adding to list
        if os.path.exists(label_full_path):
            data_pairs.append((img_full_path, label_full_path))
        else:
            print(f"Warning: Missing label file for {img_filename}")
            print(f"  - Expected: {label_full_path}")

    return data_pairs

# --- MAIN EXECUTION ---

# Define the two specific folders
images_dir = "/PHShome/yl535/project/python/datasets/BTCV/imagesTr"
labels_dir = "/PHShome/yl535/project/python/datasets/BTCV/labelsTr"

# Define output directory
output_dir = "/PHShome/yl535/project/python/datasets/BTCV/segmented" 

# Check if input directories exist to avoid crash
if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
    print("Error: Input directories not found.")
    print(f"Checked: {images_dir}")
    print(f"Checked: {labels_dir}")
else:
    # Get pairs
    pairs = get_image_mask_pairs_btcv(images_dir, labels_dir)

    print(f"\nSuccessfully loaded {len(pairs)} pairs.")

    # Process pairs
    for img_path, mask_path in pairs:
        print(f"Processing Pair: {os.path.basename(img_path)} | {os.path.basename(mask_path)}")
        segment_and_crop_objects(img_path, mask_path, output_dir=output_dir)