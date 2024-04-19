
_base_ = './faster-rcnn_r50_fpn_1x_coco.py'

# ========================Frequently modified parameters======================
# -----data related-----
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=4),
))

# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))

optim_wrapper = dict(
    optimizer=dict(lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09, history_size=100, type='LBFGS'),
    type='OptimWrapper')


dataset_type = 'COCODataset'
classes =  ('Alive Tree', 'Beetle-Fire Tree', 'Dead Tree', 'Debris')

data_root = '/work/van-speech-nlp/jindaznb/j-vis/ForestFire2023-5-log' # Root directory of the dataset

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
base_lr = 0.004  # Base learning rate for optimization
max_epochs = 500  # Maximum training epochs
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
        data_root='/work/van-speech-nlp/jindaznb/j-vis/ForestFire2023-5-log',
    ),)
test_evaluator = dict(
    ann_file='/work/van-speech-nlp/jindaznb/j-vis/ForestFire2023-5-log/valid/_annotations.coco.json',)



train_dataloader = dict(

    dataset=dict(
        data_root='/work/van-speech-nlp/jindaznb/j-vis/ForestFire2023-5-log',
    ),)

val_dataloader = dict(

    dataset=dict(

        data_root='/work/van-speech-nlp/jindaznb/j-vis/ForestFire2023-5-log',
    ),)
    
    
val_evaluator = dict(
    ann_file=
    '/work/van-speech-nlp/jindaznb/j-vis/ForestFire2023-5-log/valid/_annotations.coco.json',
)

