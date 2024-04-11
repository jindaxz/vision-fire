
_base_ = './cascade-rcnn_x101-64x4d_fpn_1x_coco.py'

# ========================Frequently modified parameters======================
# -----data related-----


dataset_type = 'COCODataset'
classes =  ('Alive Tree', 'Beetle-Fire Tree', 'Dead Tree', 'Debris')
#optimizer = dict(lr=0.02 / 10)

data_root = '/work/van-speech-nlp/jindaznb/j-vis/ForestFire2023-5' # Root directory of the dataset

train_ann_file = 'train/_annotations.coco.json'  # Annotation file for training set
train_data_prefix = 'train/'  # Prefix for training data directory

val_ann_file = 'valid/_annotations.coco.json'  # Annotation file for validation set
val_data_prefix = 'valid/'  # Prefix for validation data directory

class_name = ('Alive Tree', 'Beetle-Fire Tree', 'Dead Tree', 'Debris') 
num_classes = 4  # Number of classes in the dataset
metainfo = dict(classes=class_name, palette=[(20, 220, 60)])  # Metadata information for visualization

train_batch_size_per_gpu = 8  # Batch size per GPU during training
#train_num_workers = 4  # Number of worker processes for data loading during training
persistent_workers = True  # Whether to use persistent workers during training

# -----train val related-----
max_epochs = 500  # Maximum training epochs

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
        patience=15,
        min_delta=0.001
    ),
)

train_cfg=dict(
    max_epochs=max_epochs
)

data = dict(
    #samples_per_gpu=1,
    #workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        img_prefix='train/',
        classes=classes,
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

