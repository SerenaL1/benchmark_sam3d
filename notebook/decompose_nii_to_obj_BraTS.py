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
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data...")
    img_vol = image.load_img(intensity_nii_path)
    mask_vol = image.load_img(mask_nii_path)

    filename = os.path.basename(intensity_nii_path)
    base_name = filename.split('.nii')[0] 
    
    # Get data to find unique labels
    mask_data = mask_vol.get_fdata()
    unique_labels = np.unique(mask_data)
    # Filter out background (0)
    object_ids = unique_labels[unique_labels != 0].astype(int)
    
    print(f"Found {len(object_ids)} tumor regions: {object_ids}")
    
    for label_id in object_ids:
        print(f"Processing Tumor Region {label_id}...")
        
        # 1. Create a binary mask for the current tumor region
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
        cropped_mask = image.crop_img(binary_mask)
        cropped_object = image.resample_to_img(masked_object, cropped_mask)
        
        # 4. Save to file
        save_path_img = os.path.join(output_dir, f"{base_name}_{label_id}.nii.gz")
        save_path_mask = os.path.join(output_dir, f"{base_name}_{label_id}_mask.nii.gz")
        
        nib.save(cropped_object, save_path_img)
        nib.save(cropped_mask, save_path_mask)
        
        print(f"  Saved Image: {save_path_img} (Shape: {cropped_object.shape})")


def get_brats_image_mask_pairs(root_dir, modality="flair", use_manual=True):
    """
    Traverses BraTS dataset and finds matching MRI/Mask pairs.
    
    Args:
        root_dir: Path to Pre-operative_TCGA_LGG_NIfTI_and_Segmentations folder
        modality: Which MRI sequence to use ('flair', 't1', 't1Gd', 't2')
        use_manual: If True, prefer manually corrected masks
    """
    data_pairs = []
    
    # Get all patient folders (TCGA-XX-XXXX format)
    subfolders = [d for d in os.listdir(root_dir) 
                  if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('TCGA')]
    
    subfolders.sort()
    print(f"Found {len(subfolders)} patient folders.")

    for patient_id in subfolders:
        patient_path = os.path.join(root_dir, patient_id)
        
        # Find files in this patient folder
        files = os.listdir(patient_path)
        
        img_file = None
        mask_file = None
        manual_mask_file = None
        
        for f in files:
            if f.endswith('.nii.gz'):
                # Find the MRI scan with specified modality
                if f'_{modality}.nii.gz' in f:
                    img_file = f
                # Find GlistrBoost mask
                elif 'GlistrBoost' in f:
                    if 'ManuallyCorrected' in f:
                        manual_mask_file = f
                    else:
                        mask_file = f
        
        # Decide which mask to use (prefer manual if available and requested)
        if use_manual and manual_mask_file:
            chosen_mask = manual_mask_file
        elif mask_file:
            chosen_mask = mask_file
        elif manual_mask_file:
            chosen_mask = manual_mask_file
        else:
            chosen_mask = None
        
        if img_file and chosen_mask:
            img_full_path = os.path.join(patient_path, img_file)
            mask_full_path = os.path.join(patient_path, chosen_mask)
            data_pairs.append((img_full_path, mask_full_path))
        else:
            print(f"Warning: Missing files in folder {patient_id}")
            if not img_file: 
                print(f"  - Missing: {modality} image")
            if not chosen_mask: 
                print(f"  - Missing: GlistrBoost segmentation mask")

    return data_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", default="flair", type=str,
                        help='MRI modality to use (flair, t1, t1Gd, t2)')
    parser.add_argument("--input_dir", type=str, required=True,
                        help='Path to Pre-operative_TCGA_LGG_NIfTI_and_Segmentations folder')
    parser.add_argument("--output_dir", type=str, required=True,
                        help='Path to save segmented outputs')
    parser.add_argument("--no_manual", action='store_true',
                        help='Do not use manually corrected masks')
    args = parser.parse_args()

    # Get all image-mask pairs
    pairs = get_brats_image_mask_pairs(args.input_dir, 
                                       modality=args.modality,
                                       use_manual=not args.no_manual)
    
    print(f"\nSuccessfully found {len(pairs)} pairs.")
    
    # Process each pair
    for img_path, mask_path in pairs:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(img_path)}")
        print(f"       Mask: {os.path.basename(mask_path)}")
        print(f"{'='*60}")
        
        segment_and_crop_objects(img_path, mask_path, output_dir=args.output_dir)
    
    print(f"\n✅ Done! Output saved to: {args.output_dir}")
