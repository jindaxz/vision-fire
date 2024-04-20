#!/usr/bin/env python
# coding: utf-8

# In[7]:


import supervision as sv
import torch, torchvision
import glob
from mmdet.apis import init_detector, inference_detector
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


import argparse
import os
HOME = os.getcwd()
print("HOME:", HOME)


parser = argparse.ArgumentParser()
parser.add_argument('--model_folder', type=str, default='faster_rcnn', help='Folder for the model')
parser.add_argument('--base_model_name', type=str, default='faster-rcnn_r50_fpn_1x_coco', help='Base name for the model')
parser.add_argument('--dataset_location', type=str, default='/work/van-speech-nlp/jindaznb/j-vis/Forestfire2024-1', help='Location of the dataset')
parser.add_argument("--custom_model_name", default=f"custom-{base_model_name}")

args = parser.parse_args()
model_folder=args.model_folder
base_model_name=args.base_model_name
dataset_location=args.dataset_location


CUSTOM_CONFIG_PATH = f"{HOME}/mmdetection/configs/{model_folder}/{custom_model_name}.py"
print(CUSTOM_CONFIG_PATH)

seed_value= 42 

# 1. Set `PYTHONHASHSEED` environment variable at a fixed value

os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
# dataset_location=f"/work/van-speech-nlp/jindaznb/j-vis/Forestfire2024-1"
dataset_location=f"/work/van-speech-nlp/jindaznb/j-vis/Forestfire2024-1-lab"

# # 4. Set `tensorflow` pseudo-random generator at a fixed value
# import tensorflow as tf
# tf.random.set_seed(seed_value)


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

def optimize_params(ds, model, param_space, num_samples=30):
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


# In[8]:


CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5

ds = sv.DetectionDataset.from_coco(
    images_directory_path=f"{dataset_location}/test",
    annotations_path=f"{dataset_location}/test/_annotations.coco.json",
)

images = list(ds.images.values())
CUSTOM_WEIGHTS_PATH = glob.glob(f"{HOME}/mmdetection/work_dirs/{custom_model_name}/best_coco_bbox_mAP_50_epoch_*.pth")[-1]
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


# In[9]:


ds = sv.DetectionDataset.from_coco(
    images_directory_path=f"{dataset_location}/test",
    annotations_path=f"{dataset_location}/test/_annotations.coco.json",
)

images = list(ds.images.values())
CUSTOM_WEIGHTS_PATH = glob.glob(f"{HOME}/mmdetection/work_dirs/{custom_model_name}/best_coco_bbox_mAP_50_epoch_*.pth")[-1]
model = init_detector(CUSTOM_CONFIG_PATH, CUSTOM_WEIGHTS_PATH, device=DEVICE)

best_map, best_params = optimize_params(ds, model, param_space)
print('Best mAP:', best_map)
print('Best Parameters:', best_params)


# In[ ]:




