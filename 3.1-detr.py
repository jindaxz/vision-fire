#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import sys
os.system('nvcc -V')
os.system('gcc --version')
os.system('nvidia-smi')
os.system('pwd')
system_prefix = sys.prefix
print(f"System Prefix: {system_prefix}")

import os
HOME = os.getcwd()
print("HOME:", HOME)

# Check Pytorch installation
import cv2
import os
import json
import random

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
print(get_compiling_cuda_version())
print(get_compiler_version())

from roboflow import Roboflow
rf = Roboflow(api_key="a8fzziqukkQgsmGFnKgD")
project = rf.workspace("wildfire2024").project("forestfire2024")
dataset = project.version(1).download("coco-mmdetection")

import supervision as sv

ds = sv.DetectionDataset.from_coco(
    images_directory_path=f"{dataset.location}/train",
    annotations_path=f"{dataset.location}/train/_annotations.coco.json",
)
print('dataset classes:', ds.classes)
print('dataset size:', len(ds))

ds = sv.DetectionDataset.from_coco(
    images_directory_path=f"{dataset.location}/test",
    annotations_path=f"{dataset.location}/test/_annotations.coco.json",
)
print('dataset classes:', ds.classes)
print('dataset size:', len(ds))


# # Write custom Config file

# In[2]:


# %cd {HOME}/mmyolo
os.chdir('mmdetection')


# In[3]:


CUSTOM_CONFIG_PATH = f"{HOME}/mmdetection/configs/efficientnet/custom-retinanet_effb3_fpn_8xb4-crop896-1x_coco-Copy1.py"

CUSTOM_CONFIG = f"""
_base_ = 'retinanet_effb3_fpn_8xb4-crop896-1x_coco.py'



# ========================Frequently modified parameters======================
# -----data related-----
model = dict(
    bbox_head=dict(num_classes=4),
)

dataset_type = 'COCODataset'
max_epochs = 200
classes =  ('Alive Tree', 'Beetle-Fire Tree', 'Dead Tree', 'Debris') 

data_root = '/work/van-speech-nlp/jindaznb/j-vis/Forestfire2024-1/'  # Root directory of the dataset

train_ann_file = 'train/_annotations.coco.json'  # Annotation file for training set
train_data_prefix = 'train/'  # Prefix for training data directory

val_ann_file = 'valid/_annotations.coco.json'  # Annotation file for validation set
val_data_prefix = 'valid/'  # Prefix for validation data directory

class_name = ('Alive Tree', 'Beetle-Fire Tree', 'Dead Tree', 'Debris')  
num_classes = 4  # Number of classes in the dataset
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])  # Metadata information for visualization

train_batch_size_per_gpu = 2  # Batch size per GPU during training
train_num_workers = 4  # Number of worker processes for data loading during training
persistent_workers = True  # Whether to use persistent workers during training

# -----train val related-----
# base_lr = 0.004  # Base learning rate for optimization
base_lr = 0.04
max_epochs = 200  # Maximum training epochs
num_epochs_stage2 = 20  # Number of epochs for stage 2 training

model_test_cfg = dict(
    multi_label=True,  # Multi-label configuration for multi-class prediction
    nms_pre=30000,  # Number of boxes before NMS
    score_thr=0.001,  # Score threshold to filter out boxes
    nms=dict(type='nms', iou_threshold=0.65),  # NMS type and threshold
    max_per_img=300)  # Maximum number of detections per image

train_dataloader = dict(
     batch_size=1,
     num_workers=2,
)

val_dataloader = dict(
     batch_size=1,
     num_workers=2,
)

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
        patience=10,
        min_delta=0.001
    ),
)

train_cfg=dict(
    max_epochs=max_epochs
)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        img_prefix='train/',
        classes=classes,
        ann_file='train/annotation_coco.json'),
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
val_batch_size_per_gpu = 32
# Worker to pre-fetch data for each single GPU during validation
val_num_workers = 3

# Config of batch shapes. Only on val.
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    size_divisor=32,
    extra_pad_ratio=0.5)

# -----model related-----
# The scaling factor that controls the depth of the network structure
deepen_factor = 0.67
# The scaling factor that controls the width of the network structure
widen_factor = 0.75
# Strides of multi-scale prior box
strides = [8, 16, 32]

norm_cfg = dict(type='BN')  # Normalization config

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
"""
with open(CUSTOM_CONFIG_PATH, 'w') as file:
    file.write(CUSTOM_CONFIG)


# In[4]:


# !python tools/train.py configs/ssd/costom_ssd512.py --resume | tee ../OUT_ssd.txt
os.system('python tools/train.py configs/efficientnet/custom-retinanet_effb3_fpn_8xb4-crop896-1x_coco-Copy1.py --resume | tee "../OUT_rtemdetm_$(date +"%Y%m%d_%H%M%S").txt"')


# In[ ]:





# In[5]:


print()


# In[ ]:




