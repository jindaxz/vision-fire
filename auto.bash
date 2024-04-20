#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 3
#SBATCH -p gpu

#SBATCH --gres=gpu:t4:1   # --gres=gpu:t4:1
#SBATCH --time=08:00:00
#SBATCH --output=%j.output
#SBATCH --error=%j.error

module load anaconda3/2022.05
source activate /work/van-speech-nlp/jindaznb/visenv/

cd mmdetection


# # 41490554 #SBATCH --gres=gpu:t4:1
# model_folder="faster_rcnn"
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}-log-muti"


# 41490557
# model_folder="faster_rcnn"
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}-cbam-log-muti"


# 41490562
# model_folder="faster_rcnn"
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}-cbam-log-muti-era"


# 41715569
# model_folder="grid_rcnn"
# base_model_name="grid-rcnn_x101-64x4d_fpn_gn-head_2x_coco"
# custom_model_name="custom-${base_model_name}-2"

# 41715665
# model_folder="grid_rcnn"
# base_model_name="grid-rcnn_x101-64x4d_fpn_gn-head_2x_coco"
# custom_model_name="custom-${base_model_name}-3"

# 41618837 41637967
# model_folder="grid_rcnn"
# base_model_name="grid-rcnn_x101-64x4d_fpn_gn-head_2x_coco"
# custom_model_name="custom-${base_model_name}-mutiscale"

# 41618838
# model_folder="grid_rcnn"
# base_model_name="grid-rcnn_x101-64x4d_fpn_gn-head_2x_coco"
# custom_model_name="custom-${base_model_name}-log"

# 41684580 41715107
# model_folder="grid_rcnn"
# base_model_name="grid-rcnn_x101-64x4d_fpn_gn-head_2x_coco"
# custom_model_name="custom-${base_model_name}-log-muti"

# 41676468
# model_folder="grid_rcnn"
# base_model_name="grid-rcnn_x101-64x4d_fpn_gn-head_2x_coco"
# custom_model_name="custom-${base_model_name}-linear"

# 41618878
# model_folder="grid_rcnn"
# base_model_name="grid-rcnn_x101-64x4d_fpn_gn-head_2x_coco"
# custom_model_name="custom-${base_model_name}-random-erasing"

# 41715005
# model_folder="grid_rcnn"
# base_model_name="grid-rcnn_x101-64x4d_fpn_gn-head_2x_coco"
# custom_model_name="custom-${base_model_name}-hsv"

# 41715733 not working, indentation error. fuck.
# model_folder="grid_rcnn"
# base_model_name="grid-rcnn_x101-64x4d_fpn_gn-head_2x_coco"
# custom_model_name="custom-${base_model_name}-cbam"





# 41901040 41901150 41914242
# model_folder="grid_rcnn"
# base_model_name="grid-rcnn_x101-64x4d_fpn_gn-head_2x_coco"
# custom_model_name="custom-${base_model_name}-log-muti"

# 41914248
# model_folder="yolox"
# base_model_name="yolox_x_8xb8-300e_coco"
# custom_model_name="custom-${base_model_name}"
 
# 41914249
# model_folder="yolox"
# base_model_name="yolox_x_8xb8-300e_coco"
# custom_model_name="custom-${base_model_name}-multi"

# 41914250
# model_folder="yolox"
# base_model_name="yolox_x_8xb8-300e_coco"
# custom_model_name="custom-${base_model_name}-log"

# 41914252
model_folder="yolox"
base_model_name="yolox_x_8xb8-300e_coco"
custom_model_name="custom-${base_model_name}-log-multi"







# 41715770
# model_folder="mask_rcnn"
# base_model_name="mask-rcnn_x101-64x4d_fpn_ms-poly_3x_coco.py"
# custom_model_name=f"custom-${base_model_name}"



### repeated 3 rounds
# 41715777
# model_folder="faster_rcnn"
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}-mutiscale-2"

# 41715780
# model_folder="faster_rcnn"
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}-mutiscale-3"

# 41715785
# model_folder="faster_rcnn"
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}-log-2"

# 41715788
# model_folder="faster_rcnn"
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}-log-3"

# 41715794
# model_folder="faster_rcnn"
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}-log-muti-2"

# 41715795
# model_folder="faster_rcnn"
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# custom_model_name="custom-${base_model_name}-log-muti-3"





python tools/train.py \
    configs/${model_folder}/${custom_model_name}.py \
    --resume\
    --cfg-options randomness.seed=42 \
    | tee "../log/OUT_${custom_model_name}_$(date +"%Y%m%d_%H%M%S").txt"
    
# python tools/train.py \
#     configs/${model_folder}/${custom_model_name}.py \
#     --cfg-options randomness.seed=42 \
#     | tee "../log/OUT_${custom_model_name}_$(date +"%Y%m%d_%H%M%S").txt"
    