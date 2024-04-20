#!/bin/bash -l
#SBATCH -N 1
#SBATCH -c 3
#SBATCH -p gpu
#SBATCH --gres=gpu:1   # Use any available GPU
#SBATCH --time=03:00:00
#SBATCH --output=%j.output
#SBATCH --error=%j.error

module load anaconda3/2022.05
source activate /work/van-speech-nlp/jindaznb/visenv/



# 41768422 41800959
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# python auto_inference.py \
#     --model_folder "faster_rcnn" \
#     --base_model_name "${base_model_name}" \
#     --custom_model_name "custom-${base_model_name}" \
#     --dataset_location "/work/van-speech-nlp/jindaznb/j-vis/ForestFire2023-5"
 

# 41801153 41805683
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# python auto_inference.py \
#     --model_folder "faster_rcnn" \
#     --base_model_name "${base_model_name}" \
#     --custom_model_name "custom-${base_model_name}-mutiscale" \
#     --dataset_location "/work/van-speech-nlp/jindaznb/j-vis/ForestFire2023-5"


# 41741283 41800960
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# python auto_inference.py \
#     --model_folder "faster_rcnn" \
#     --base_model_name "${base_model_name}" \
#     --custom_model_name "custom-${base_model_name}-cbam"\
#     --dataset_location "/work/van-speech-nlp/jindaznb/j-vis/ForestFire2023-5"

# 41800965 41805679
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# python auto_inference.py \
#     --model_folder "faster_rcnn" \
#     --base_model_name "${base_model_name}" \
#     --custom_model_name "custom-${base_model_name}-random-erasing"\
#     --dataset_location "/work/van-speech-nlp/jindaznb/j-vis/ForestFire2023-5"

# 41800967
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# python auto_inference.py \
#     --model_folder "faster_rcnn" \
#     --base_model_name "${base_model_name}" \
#     --custom_model_name "custom-${base_model_name}-yuv"\
#     --dataset_location "/work/van-speech-nlp/jindaznb/j-vis/ForestFire2023-5-yuv"

# 41800968
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# python auto_inference.py \
#     --model_folder "faster_rcnn" \
#     --base_model_name "${base_model_name}" \
#     --custom_model_name "custom-${base_model_name}-lab"\
#     --dataset_location "/work/van-speech-nlp/jindaznb/j-vis/ForestFire2023-5-lab"


# 41800970
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# python auto_inference.py \
#     --model_folder "faster_rcnn" \
#     --base_model_name "${base_model_name}" \
#     --custom_model_name "custom-${base_model_name}-log"\
#     --dataset_location "/work/van-speech-nlp/jindaznb/j-vis/ForestFire2023-5-log"

# 41800971
# base_model_name="faster-rcnn_r50_fpn_1x_coco"
# python auto_inference.py \
#     --model_folder "faster_rcnn" \
#     --base_model_name "${base_model_name}" \
#     --custom_model_name "custom-${base_model_name}-linear"\
#     --dataset_location "/work/van-speech-nlp/jindaznb/j-vis/ForestFire2023-5-linear"



