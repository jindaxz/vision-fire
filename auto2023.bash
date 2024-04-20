#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 3
#SBATCH -p gpu

#SBATCH --gres=gpu:k80:1
#SBATCH --time=08:00:00
#SBATCH --output=%j.output
#SBATCH --error=%j.error

module load anaconda3/2022.05
source activate /work/van-speech-nlp/jindaznb/visenv/

cd mmdetection

# --gres=gpu:t4:1

# 41856157
# model_folder="faster_rcnn"
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}"

# 41856209
model_folder="faster_rcnn"
base_model_name="faster-rcnn_r50_fpn_1x_coco"
custom_model_name="custom-${base_model_name}-cbam"

# 41755869 41784506
# model_folder="faster_rcnn"
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}-random-erasing"

# 41755871 41784507
# model_folder="faster_rcnn"
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}-mutiscale"

# 41755872 41784509
# model_folder="faster_rcnn"
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}-lab"

# 41755873 41784510
# model_folder="faster_rcnn"
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}-linear"

# 41784511
# model_folder="faster_rcnn"
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}-log"

# 41854548
# model_folder="faster_rcnn"
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}-log-adamW"

# 41801195 41805750 41854684
# model_folder="faster_rcnn"
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}-log-rms"



# 41744429 41784512
# model_folder="faster_rcnn"
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}-yuv"

# 41744431 41784513
# model_folder="faster_rcnn"
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}-log-muti"

# 41744437 41784514
# model_folder="faster_rcnn"
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}-log-random"




# 41744444
# model_folder="dynamic_rcnn"
# base_model_name="dynamic-rcnn_r50_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}"


# 41768419 41784449
# model_folder="grid_rcnn"
# base_model_name="grid-rcnn_x101-64x4d_fpn_gn-head_2x_coco"              
# custom_model_name="custom-${base_model_name}"

# 41744447 41784516 41800954
# model_folder="libra_rcnn"
# base_model_name="libra-faster-rcnn_x101-64x4d_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}"

# 41784518
# model_folder="sparse_rcnn"
# base_model_name="sparse-rcnn_r101_fpn_300-proposals_crop-ms-480-800-3x_coco"
# custom_model_name="custom-${base_model_name}"

# 41744464 41784520
# model_folder="cascade_rcnn"
# base_model_name="cascade-rcnn_x101-64x4d_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}"


python tools/train.py \
    configs/${model_folder}/${custom_model_name}.py \
    --resume\
    --cfg-options randomness.seed=42 \
    | tee "../log/OUT_${custom_model_name}_$(date +"%Y%m%d_%H%M%S").txt"
    
# python tools/train.py \
#     configs/${model_folder}/${custom_model_name}.py \
#     --cfg-options randomness.seed=42 \
#     | tee "../log/OUT_${custom_model_name}_$(date +"%Y%m%d_%H%M%S").txt"
    