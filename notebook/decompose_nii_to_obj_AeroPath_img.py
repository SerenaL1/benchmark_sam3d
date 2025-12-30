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

# def visualize_segmented_object(segmented_slice, label_id):
#     """
#     Function 3: Visualizes the segmented object.
#     """
#     # Rotate for standard orientation
#     to_show = np.rot90(segmented_slice)
    
#     plt.figure(figsize=(4, 4))
#     plt.imshow(to_show, cmap='gray')
#     plt.title(f"Object Label: {label_id}")
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()

type = "lungs"  # "airways" or "lungs"
input_folder = "/PHShome/yl535/project/python/datasets/AeroPath/data" 
output_dir = f"/PHShome/yl535/project/python/datasets/AeroPath/{type}_segmented_img" 

pairs = get_image_mask_pairs(input_folder, type=type)

print(f"\nSuccessfully loaded {len(pairs)} pairs.")
os.makedirs(output_dir, exist_ok=True)

# Example loop to process them
for img_path, mask_path in pairs:
    filename = os.path.basename(img_path)
    base_name = filename.split('.nii')[0] 

    print(f"Processing: {os.path.basename(img_path)}, {os.path.basename(mask_path)}")

    # segment_and_crop_objects(img_path, mask_path, output_dir=output_dir)

    extracted_objects = extract_all_objects_middle_slices(img_path, mask_path, axis=2)
    for label_id, slice_data in extracted_objects.items():
        print(label_id)
        print(type(slice_data))
        save_path = os.path.join(output_dir, f"{base_name}_{label_id:02d}.nii.gz")
        nib.save(slice_data, save_path)