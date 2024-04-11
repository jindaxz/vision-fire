# The new config inherits a base config to highlight the necessary modification
_base_ = 'grounding_dino_swin-l_pretrain_all.py'


# We also need to change the num_classes in head to match the dataset's annotation
# model = dict(
#     roi_head=dict(
#         bbox_head=dict(num_classes=4),
#         mask_head=dict(num_classes=4)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes =  ('Alive Tree', 'Beetle-Fire Tree', 'Dead Tree', 'Debris') 
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
        type='CheckpointHook',
        interval=1,
        save_best='coco/bbox_mAP'
    ),
    early_stopping=dict(
        type="EarlyStoppingHook",
        monitor="coco/bbox_mAP",
        patience=10,
        min_delta=0.005
    ),
)