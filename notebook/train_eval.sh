#!/bin/bash

source activate base
conda activate sam3d-objects

CONFIG_PATH="/PHShome/yl535/project/python/sam_3d/sam-3d-objects/checkpoints/checkpoints/pipeline.yaml"
FILE_SUFFIX=".nii.gz"

# SLICE_AXIS=2
# PRED_DIR="/PHShome/yl535/project/python/sam_3d/sam-3d-objects/results_AeroPath/lungs_${SLICE_AXIS}/"
# GT_DIR="/PHShome/yl535/project/python/datasets/AeroPath/lungs_segmented/"

SLICE_AXIS=2
PRED_DIR="/PHShome/yl535/project/python/sam_3d/sam-3d-objects/results_BTCV/axis_${SLICE_AXIS}/"
GT_DIR="/PHShome/yl535/project/python/datasets/BTCV/segmented/"


# for (( i=0; i<${#GEN_IMAGE_DIR[@]}; i++ ));               

python inference_AeroPath.py \
    --config_path ${CONFIG_PATH} \
    --input_folder ${GT_DIR} \
    --output_folder ${PRED_DIR} \
    --slice_axis ${SLICE_AXIS}

python evaluate_shape_metrics.py \
    --pred_folder ${PRED_DIR} \
    --gt_folder ${GT_DIR} \
    --file_suffix ${FILE_SUFFIX}