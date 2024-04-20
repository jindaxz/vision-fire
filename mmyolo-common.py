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
# rf = Roboflow(api_key="a8fzziqukkQgsmGFnKgD")
# project = rf.workspace("wildfire2024").project("forestfire2024")
# dataset = project.version(1).download("coco-mmdetection")

import supervision as sv



# # SET RANDOMNESS

# In[1]:


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


# # Bayesian OPTIM

# In[ ]:


# def random_search_for_best_params(ds, param_space, model, num_samples=30):
#     best_map = 0
#     best_params = []

#     for _ in range(num_samples):
#         CONFIDENCE_THRESHOLD = np.random.choice(param_space['CONFIDENCE_THRESHOLD'])
#         NMS_IOU_THRESHOLD = np.random.choice(param_space['NMS_IOU_THRESHOLD'])

#         def callback(image: np.ndarray) -> sv.Detections:
#             result = inference_detector(model, image)
#             detections = sv.Detections.from_mmdetection(result)
#             return detections[detections.confidence > CONFIDENCE_THRESHOLD].with_nms(threshold=NMS_IOU_THRESHOLD)

#         mean_average_precision = sv.MeanAveragePrecision.benchmark(dataset=ds, callback=callback)

#         if mean_average_precision.map50_95 > best_map:
#             best_map = mean_average_precision.map50_95
#             best_params = [CONFIDENCE_THRESHOLD, NMS_IOU_THRESHOLD]

#     return best_map, best_params

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

    optimizer.maximize(init_points=5, n_iter=num_samples - 5)  

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

