
# The new config inherits a base config to highlight the necessary modification
_base_ = 'grid-rcnn_x101-64x4d_fpn_gn-head_2x_coco.py'

# ========================Frequently modified parameters======================

dataset_type = 'COCODataset'
max_epochs = 200
classes =  ('Alive Tree', 'Beetle-Fire Tree', 'Dead Tree', 'Debris') 

data_root = '/work/van-speech-nlp/jindaznb/j-vis/Forestfire2024-1'

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


train_dataloader = dict(
    batch_size=4,
)

val_dataloader = dict(
    batch_size=4,
)

test_dataloader = dict(
    batch_size=4,
)

