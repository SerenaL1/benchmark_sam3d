import numpy as np
import os
from nilearn import image
import nibabel as nib

def segment_and_crop_objects(intensity_nii_path, mask_nii_path, output_dir="segmented_objects"):
    """
    Splits a multi-label NIfTI mask into individual segmented NIfTI files.
    Applies the mask to the intensity image and crops to the object's bounding box.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data...")
    img_vol = image.load_img(intensity_nii_path)
    mask_vol = image.load_img(mask_nii_path)

    # --- MODIFICATION START: Extract base name ---
    # Get filename "image.nii.gz", then strip extensions to get "image"
    filename = os.path.basename(intensity_nii_path)
    # Split at .nii to handle both .nii and .nii.gz robustly
    base_name = filename.split('.nii')[0] 
    # --- MODIFICATION END ---
    
    # Get data to find unique labels
    mask_data = mask_vol.get_fdata()
    unique_labels = np.unique(mask_data)
    # Filter out background (0)
    object_ids = unique_labels[unique_labels != 0].astype(int)
    
    print(f"Found {len(object_ids)} objects: {object_ids}")
    
    for label_id in object_ids:
        print(f"Processing Object {label_id}...")
        
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
        cropped_object = image.crop_img(masked_object)
        
        # 4. Save to file
        # --- MODIFICATION START: Update naming convention ---
        # Format: image_[object_id].nii.gz
        save_path = os.path.join(output_dir, f"{base_name}_{label_id}.nii.gz")
        # --- MODIFICATION END ---
        
        nib.save(cropped_object, save_path)
        print(f"  Saved to: {save_path} (Shape: {cropped_object.shape})")

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

type = "lungs"  # "airways" or "lungs"
input_folder = "/PHShome/yl535/project/python/datasets/AeroPath/data" 
output_dir = f"/PHShome/yl535/project/python/datasets/AeroPath/{type}_segmented" 

pairs = get_image_mask_pairs(input_folder, type=type)

print(f"\nSuccessfully loaded {len(pairs)} pairs.")

# Example loop to process them
for img_path, mask_path in pairs:
    filename = os.path.basename(img_path)
    base_name = filename.split('.nii')[0] 

    print(f"Processing: {os.path.basename(img_path)}, {os.path.basename(mask_path)}")

    segment_and_crop_objects(img_path, mask_path, output_dir=output_dir)