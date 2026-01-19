#!/bin/bash

source activate base
conda activate sam3d-objects

# CORRECT checkpoint path (no /checkpoints/checkpoints)
CONFIG_PATH="$HOME/benchmark_sam3d/checkpoints/hf/pipeline.yaml"
FILE_SUFFIX=".nii.gz"

# Your BraTS data - QUOTED because of spaces in path!
SLICE_AXIS=2  # axial slices (standard for brain imaging)
PRED_DIR="$HOME/benchmark_sam3d/results_BraTS/axis_${SLICE_AXIS}/"
GT_DIR="/home/serena-liu/Downloads/PKG - BraTS-TCGA-LGG/brats_segmented/"

echo "=========================================="
echo "BraTS SAM3D Pipeline"
echo "=========================================="
echo "Config:  ${CONFIG_PATH}"
echo "Input:   ${GT_DIR}"
echo "Output:  ${PRED_DIR}"
echo "Axis:    ${SLICE_AXIS}"
echo "=========================================="

# Run SAM3D inference - ADD QUOTES AROUND VARIABLES!
python inference_BraTS.py \
    --config_path "${CONFIG_PATH}" \
    --input_folder "${GT_DIR}" \
    --output_folder "${PRED_DIR}" \
    --slice_axis ${SLICE_AXIS}

# Evaluate results - ADD QUOTES AROUND VARIABLES!
python evaluate_shape_metrics.py \
    --pred_folder "${PRED_DIR}" \
    --gt_folder "${GT_DIR}" \
    --file_suffix "${FILE_SUFFIX}"