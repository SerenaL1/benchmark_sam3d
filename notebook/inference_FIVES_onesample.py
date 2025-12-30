import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

import imageio
import uuid
from IPython.display import Image as ImageDisplay
from inference import Inference, ready_gaussian_for_video_rendering, render_video, load_image, load_single_mask, display_image, make_scene, interactive_visualizer

import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from transformers import pipeline
from io import BytesIO

def resize_pair(image, mask, target_width=512, target_height=512):
    """
    Resizes a fundus image and its corresponding mask to a specific resolution.
    
    Args:
        image (np.array): The RGB fundus photo.
        mask (np.array): The binary mask (uint8 or bool).
        target_width (int): Desired width.
        target_height (int): Desired height.
        
    Returns:
        tuple: (resized_image, resized_mask)
    """
    dsize = (target_width, target_height)
    
    # 1. Resize Image
    # cv2.INTER_AREA is best for downscaling (e.g. 2000x2000 -> 512x512) to avoid aliasing.
    # cv2.INTER_LINEAR is standard for upscaling.
    img_resized = cv2.resize(image, dsize, interpolation=cv2.INTER_AREA)
    
    # 2. Resize Mask
    # MUST use cv2.INTER_NEAREST to preserve binary values (0/1). 
    # Other methods will introduce gradients at the edges.
    mask_to_resize = mask.astype(np.uint8) # Ensure it's numeric for OpenCV
    mask_resized = cv2.resize(mask_to_resize, dsize, interpolation=cv2.INTER_NEAREST)
    
    # If your input was boolean, cast it back
    if mask.dtype == bool:
        mask_resized = mask_resized.astype(bool)
        
    return img_resized, mask_resized

def load_mask_as_boolean(mask_path, threshold=127):
    """
    Loads a mask image file and converts it to a boolean matrix.

    Args:
        mask_path (str): Path to the image file (e.g., .png, .jpg).
        threshold (int): Pixel value cutoff. Values > threshold become True. 
                         Default 127 handles standard 0 vs 255 binary masks well.

    Returns:
        np.ndarray: A boolean matrix (True/False) with the same height/width as the file.
    """
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    # 1. Read the image as Grayscale
    # This ensures we get a 2D array (H, W) immediately, ignoring color channels.
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask_img is None:
        raise ValueError(f"Failed to decode image file: {mask_path}")

    # 2. Convert to Boolean
    # We use a threshold comparison rather than direct casting (.astype(bool))
    # to be safe against JPEG compression artifacts (where 0 might become 1 or 2).
    mask_bool = mask_img > threshold

    return mask_bool

def get_boolean_mask_from_contour(contour_path, height, width):
    """
    Generates a Boolean mask (True/False) from a contour text file.
    
    Returns:
        np.ndarray: A boolean matrix of shape (height, width).
                    True indicates the region inside the contour.
    """
    # 1. Load data
    try:
        contour_data = np.loadtxt(contour_path)
    except Exception as e:
        raise IOError(f"Could not read contour file: {e}")

    # 2. Prepare points for OpenCV (requires integer coordinates)
    points = contour_data.astype(np.int32)
    points = points.reshape((-1, 1, 2))

    # 3. Create a temporary numeric mask (OpenCV needs uint8, not bool)
    temp_mask = np.zeros((height, width), dtype=np.uint8)
    
    # 4. Fill the polygon with a non-zero value (e.g., 1 or 255)
    cv2.fillPoly(temp_mask, [points], color=1)

    # 5. Convert to Boolean
    # Any non-zero value becomes True, 0 becomes False
    boolean_mask = temp_mask.astype(bool)

    return boolean_mask

def get_mask_from_contour(contour_path, height, width):
    """
    Helper function: Reads contour coordinates and generates a binary mask.
    """
    # 1. Load data
    try:
        contour_data = np.loadtxt(contour_path)
    except Exception as e:
        raise IOError(f"Could not read contour file: {e}")

    # 2. Check data validity
    if contour_data.ndim != 2 or contour_data.shape[1] != 2:
        raise ValueError("Contour file must contain 2 columns (x, y).")

    # 3. Convert to integer points for OpenCV
    points = contour_data.astype(np.int32)
    points = points.reshape((-1, 1, 2))

    # 4. Create and fill mask
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], color=255)

    return mask

def process_fundus_contour(image_path, contour_path, show_plot=True):
    """
    Main driver: Loads the image, calls the mask generator, and visualizes the result.
    """
    # 1. Load the Image
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return None, None

    # Load image and convert BGR -> RGB for Matplotlib
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("Error: Failed to decode image.")
        return None, None
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    # 2. Get the Mask (Refactored call)
    try:
        mask = get_mask_from_contour(contour_path, h, w)
    except Exception as e:
        print(f"Error generating mask: {e}")
        return None, None

    # 3. Visualization
    if show_plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot Original
        axes[0].imshow(img_rgb)
        axes[0].set_title("Original Fundus Image")
        axes[0].axis('off')
        
        # Plot Mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Binary Mask")
        axes[1].axis('off')
        
        # Plot Overlay
        axes[2].imshow(img_rgb)
        axes[2].imshow(mask, cmap='Greens', alpha=0.4) # Green overlay
        
        # Optional: Plot the specific contour line
        contour_data = np.loadtxt(contour_path)
        # Close the loop for the line plot
        x_plot = np.append(contour_data[:, 0], contour_data[0, 0])
        y_plot = np.append(contour_data[:, 1], contour_data[0, 1])
        axes[2].plot(x_plot, y_plot, 'r--', linewidth=1.5)
        
        axes[2].set_title("Segmentation Overlay")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

    # return img_rgb, mask


PATH = os.getcwd()
TAG = "hf"
config_path = f"/shared/ssd_28T/home/yl535/project/python/sam_3d/sam-3d-objects/checkpoints/checkpoints/pipeline.yaml"
# config_path = f"{PATH}/../checkpoints/checkpoints/pipeline.yaml"
inference = Inference(config_path, compile=False)

device = 0 if torch.cuda.is_available() else -1
generator = pipeline(
    "mask-generation", 
    model="facebook/sam2-hiera-large", 
    device=device,
    torch_dtype=torch.float32 
)

type="cup" # "cup" or "disc"
input_img = "/PHShome/yl535/project/python/datasets/Fundus_Image_Vessel_Segmentation/test/Original/136_G.png"
input_mask = "/PHShome/yl535/project/python/datasets/Fundus_Image_Vessel_Segmentation/test/Ground_truth/136_G.png"
output_dir = f"/shared/ssd_28T/home/yl535/project/python/sam_3d/sam-3d-objects/results/FIVES/gaussians/"

filename = os.path.basename(input_img)
base_name = filename.split('.png')[0] 

img_bgr = cv2.imread(input_img)
if img_bgr is None:
    print("Error: Failed to decode image.")
    
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w, _ = img_rgb.shape

# 2. Get the Mask (Refactored call)
try:
    mask = load_mask_as_boolean(input_mask)
except Exception as e:
    print(f"Error generating mask: {e}")

os.makedirs(output_dir, exist_ok=True)
# run model
rz_img, rz_mask = resize_pair(img_rgb, mask)
output = inference(rz_img, rz_mask, seed=42)

# export gaussian splat (as point cloud)
output["gs"].save_ply(os.path.join(output_dir, f"{base_name}.ply"))

# render gaussian splat
scene_gs = make_scene(output)
scene_gs = ready_gaussian_for_video_rendering(scene_gs, fix_alignment=True)

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
    os.path.join(output_dir, f"{base_name}.gif"),
    video,
    format="GIF",
    duration=1000 / 30,  # default assuming 30fps from the input MP4
    loop=0,  # 0 means loop indefinitely
)