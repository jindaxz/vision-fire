#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
system_prefix = sys.prefix
print(f"System Prefix: {system_prefix}")

import os
HOME = os.getcwd()
print("HOME:", HOME)

import os
import json
import random
import glob
import argparse


import numpy as np

from mmdet.apis import init_detector, inference_detector

import torch, torchvision
print(torch.__version__, torch.cuda.is_available())
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Check MMDetection installation
from mmengine.utils import get_git_hash
from mmengine.utils.dl_utils import collect_env as collect_base_env

import mmdet

print(mmdet.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
from roboflow import Roboflow

import supervision as sv



seed_value= 42 

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
dataset_location=f"/work/van-speech-nlp/jindaznb/j-vis/Forestfire2024-1"

# # 4. Set `tensorflow` pseudo-random generator at a fixed value
# import tensorflow as tf
# tf.random.set_seed(seed_value)


# In[2]:


param_space = {
    'CONFIDENCE_THRESHOLD': np.linspace(0.1, 1.0, 20),  # Adjust the range as needed
    'NMS_IOU_THRESHOLD': np.linspace(0.1, 1.0, 20),  # Adjust the range as needed
}

best_map = 0.0
best_params = {}

MAX_EPOCHS=300
BATCH_SIZE=8
classes=('Alive Tree', 'Beetle-Fire Tree', 'Dead Tree', 'Debris')
num_classes=len(classes)




from bayes_opt import BayesianOptimization

def optimize_params(ds, model, param_space, num_samples=50):
    def objective_function(confidence_threshold, nms_iou_threshold):
        def callback(image: np.ndarray) -> sv.Detections:
            result = inference_detector(model, image)
            detections = sv.Detections.from_mmdetection(result)
            return detections[detections.confidence > confidence_threshold].with_nms(threshold=nms_iou_threshold)

        mean_average_precision = sv.MeanAveragePrecision.benchmark(dataset=ds, callback=callback)
        return mean_average_precision.map50_95

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds={'confidence_threshold': (param_space['CONFIDENCE_THRESHOLD'][0], param_space['CONFIDENCE_THRESHOLD'][-1]),
                 'nms_iou_threshold': (param_space['NMS_IOU_THRESHOLD'][0], param_space['NMS_IOU_THRESHOLD'][-1])},
        random_state=42,  
    )

    optimizer.maximize(init_points=5, n_iter=num_samples)  

    best_params = [optimizer.max['params']['confidence_threshold'], optimizer.max['params']['nms_iou_threshold']]
    best_map = optimizer.max['target']

    return best_map, best_params


# In[ ]:


def calculate_mAP_default(dataset_location: str, model_name: str, custom_config_path: str, device: str = 'cuda', confidence_threshold: float = 0.5, nms_iou_threshold: float = 0.5) -> float:
    ds = sv.DetectionDataset.from_coco(
        images_directory_path=f"{dataset_location}/test",
        annotations_path=f"{dataset_location}/test/_annotations.coco.json",
    )

    images = list(ds.images.values())
    custom_weights_path = glob.glob(f"{HOME}/mmdetection/work_dirs/{model_name}/best_coco_bbox_mAP_50_epoch_*.pth")[-1]
    model = init_detector(custom_config_path, custom_weights_path, device=device)

    def callback(image: np.ndarray) -> sv.Detections:
        result = inference_detector(model, image)
        detections = sv.Detections.from_mmdetection(result)
        return detections[detections.confidence > confidence_threshold].with_nms(threshold=nms_iou_threshold)

    mean_average_precision = sv.MeanAveragePrecision.benchmark(
        dataset=ds,
        callback=callback
    )

    return mean_average_precision.map50_95




# # Write custom Config file

# In[2]:


import os

# Define the directory path
directory = os.path.join(HOME, 'mmdetection')

# Change the current working directory
os.chdir(directory)


# In[3]:
HOME = os.getcwd()
print("HOME:", HOME)
MAX_EPOCHS=500
BATCH_SIZE=8
classes=('Alive Tree', 'Beetle-Fire Tree', 'Dead Tree', 'Debris')
num_classes=len(classes)

'''
args
'''
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--model_folder', type=str, default='faster_rcnn', help='Description of model_folder')
parser.add_argument('--base_model_name', type=str, default='faster-rcnn_r50_fpn_1x_coco', help='Description of base_model_name')
parser.add_argument('--custom_model_name', type=str, help='Description of custom_model_name')
parser.add_argument('--dataset_location', type=str, default="/work/van-speech-nlp/jindaznb/j-vis/ForestFire2023-5",
                    help='Location of the dataset')

args = parser.parse_args()

# Access the values using args
model_folder = args.model_folder
base_model_name = args.base_model_name
custom_model_name = args.custom_model_name if args.custom_model_name else f"custom-{args.base_model_name}-log-muti"
dataset_location = args.dataset_location

print(f"Model Folder: {model_folder}")
print(f"Base Model Name: {base_model_name}")
print(f"Custom Model Name: {custom_model_name}")



CUSTOM_CONFIG_PATH = f"{HOME}/configs/{model_folder}/{custom_model_name}.py"
print(CUSTOM_CONFIG_PATH)


CUSTOM_CONFIG = f"""
_base_ = './{base_model_name}.py'

# ========================Frequently modified parameters======================
# -----data related-----
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes={num_classes}),
))

dataset_type = 'COCODataset'
classes =  {classes}

data_root = '{dataset_location}' # Root directory of the dataset

train_ann_file = 'train/_annotations.coco.json'  # Annotation file for training set
train_data_prefix = 'train/'  # Prefix for training data directory

val_ann_file = 'valid/_annotations.coco.json'  # Annotation file for validation set
val_data_prefix = 'valid/'  # Prefix for validation data directory

class_name = {classes} 
num_classes = {num_classes}  # Number of classes in the dataset
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])  # Metadata information for visualization

train_batch_size_per_gpu = {BATCH_SIZE}  # Batch size per GPU during training
#train_num_workers = 4  # Number of worker processes for data loading during training
persistent_workers = True  # Whether to use persistent workers during training

# -----train val related-----
base_lr = 0.004  # Base learning rate for optimization
max_epochs = {MAX_EPOCHS}  # Maximum training epochs
num_epochs_stage2 = 20  # Number of epochs for stage 2 training

model_test_cfg = dict(
    multi_label=True,  # Multi-label configuration for multi-class prediction
    nms_pre=30000,  # Number of boxes before NMS
    score_thr=0.001,  # Score threshold to filter out boxes
    nms=dict(type='nms', iou_threshold=0.65),  # NMS type and threshold
    max_per_img=300)  # Maximum number of detections per image


# ========================Possible modified parameters========================
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        save_best="coco/bbox_mAP_50",
        rule="greater",
        max_keep_ckpts=10,
    ),
    early_stopping=dict(
        type="EarlyStoppingHook",
        monitor="coco/bbox_mAP_50",
        patience=20,
        min_delta=0.001
    ),
)

train_cfg=dict(
    max_epochs=max_epochs
)

data = dict(
    samples_per_gpu=8,
    #workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        img_prefix='train/',
        classes=classes,
        img_scale=[(1333, 800), (1666, 1000)], # mutiscale
        ann_file='train/_annotations.coco.json.json'),
    val=dict(
        type=dataset_type,
        img_prefix='valid/',
        classes=classes,
        ann_file='valid/_annotations.coco.json'),
    test=dict(
        type=dataset_type,
        img_prefix='test/',
        classes=classes,
        ann_file='test/_annotations.coco.json'))

# -----data related-----
img_scale = (1024, 1024)  # width, height
# ratio range for random resize
random_resize_ratio_range = (0.1, 2.0)
# Cached images number in mosaic
mosaic_max_cached_images = 40
# Number of cached images in mixupep
mixup_max_cached_images = 20
# Batch size of a single GPU during validation
val_batch_size_per_gpu = 8
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 3

# Config of batch shapes. Only on val.
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    size_divisor=32,
    extra_pad_ratio=0.5)

# -----train val related-----
lr_start_factor = 1.0e-5
dsl_topk = 13  # Number of bbox selected in each level
loss_cls_weight = 1.0
loss_bbox_weight = 2.0
qfl_beta = 2.0  # beta of QualityFocalLoss
weight_decay = 0.05

# Save model checkpoint and validation intervals
save_checkpoint_intervals = 10
# validation intervals in stage 2
val_interval_stage2 = 1
# The maximum checkpoints to keep.
max_keep_ckpts = 3
# single-scale training is recommended to
# be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)



test_dataloader = dict(
    dataset=dict(
        data_root='{dataset_location}',
    ),)
test_evaluator = dict(
    ann_file='{dataset_location}/valid/_annotations.coco.json',)



train_dataloader = dict(

    dataset=dict(
        data_root='{dataset_location}',
    ),)

val_dataloader = dict(

    dataset=dict(

        data_root='{dataset_location}',
    ),)
    
    
val_evaluator = dict(
    ann_file=
    '{dataset_location}/valid/_annotations.coco.json',
)

"""
with open(CUSTOM_CONFIG_PATH, 'w') as file:
    file.write(CUSTOM_CONFIG)


# In[4]:


# !python tools/train.py configs/{model_folder}/{custom_model_name}.py \
#     --resume --cfg-options randomness.seed=42 \
#     | tee "../log/OUT_{custom_model_name}_$(date +"%Y%m%d_%H%M%S").txt"


# In[5]:


CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5

ds = sv.DetectionDataset.from_coco(
    images_directory_path=f"{dataset_location}/test",
    annotations_path=f"{dataset_location}/test/_annotations.coco.json",
)

images = list(ds.images.values())
CUSTOM_WEIGHTS_PATH = glob.glob(f"{HOME}/work_dirs/{custom_model_name}/best_coco_bbox_mAP_50_epoch_*.pth")[0]
model = init_detector(CUSTOM_CONFIG_PATH, CUSTOM_WEIGHTS_PATH, device=DEVICE)

def callback(image: np.ndarray) -> sv.Detections:
    result = inference_detector(model, image)
    detections = sv.Detections.from_mmdetection(result)
    return detections[detections.confidence > CONFIDENCE_THRESHOLD].with_nms(threshold=NMS_IOU_THRESHOLD)


mean_average_precision = sv.MeanAveragePrecision.benchmark(
    dataset = ds,
    callback = callback
)

print('mAP:', mean_average_precision.map50_95)


# In[ ]:


ds = sv.DetectionDataset.from_coco(
    images_directory_path=f"{dataset_location}/test",
    annotations_path=f"{dataset_location}/test/_annotations.coco.json",
)

images = list(ds.images.values())
CUSTOM_WEIGHTS_PATH = glob.glob(f"{HOME}/work_dirs/{custom_model_name}/best_coco_bbox_mAP_50_epoch_*.pth")[0]
model = init_detector(CUSTOM_CONFIG_PATH, CUSTOM_WEIGHTS_PATH, device=DEVICE)

best_map, best_params = optimize_params(ds, model, param_space)
print('Best mAP:', best_map)
print('Best Parameters:', best_params)


import os

# Define the path for the log file
log_dir = 'log'
os.makedirs(log_dir, exist_ok=True)  # Create the log directory if it doesn't exist
log_file_path = os.path.join(log_dir, 'results_{custom_model_name}.txt')

# Open the log file in write mode
with open(log_file_path, 'w') as log_file:
    # Write the mean average precision to the log file
    log_file.write(f'mAP 0.25-0.9: {mean_average_precision.map50_95}\n')
    
    # Write the best mAP and best parameters (if available) to the log file
    if best_params is not None:
        log_file.write(f'Best mAP 0.25-0.9: {best_map}\n')
        log_file.write(f'Best Parameters: {best_params}\n')
