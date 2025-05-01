# the new config inherits the base configs to highlight the necessary modification
_base_ = '/teamspace/studios/this_studio/mmrotate/demo/rotated_faster_rcnn_r50_fpn_1x_dota_le90_val.py'
work_dir = '/teamspace/studios/this_studio/mmrotate/demo/rotated_frcnn_4_lr'

# 1. dataset settings
dataset_type = 'DOTADataset'
classes =  ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane')
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/teamspace/studios/this_studio/mmrotate/data/DOTA1_5/train/annfiles_corr/',
        img_prefix='/teamspace/studios/this_studio/mmrotate/data/DOTA1_5/train/images/'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/teamspace/studios/this_studio/mmrotate/data/DOTA1_5/val/annfiles_corr/',
        img_prefix='/teamspace/studios/this_studio/mmrotate/data/DOTA1_5/val/images/'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/teamspace/studios/this_studio/mmrotate/data/DOTA1_5/val/annfiles_corr/',
        img_prefix='/teamspace/studios/this_studio/mmrotate/data/DOTA1_5/val/images/'))

# 2. model settings
model = dict(
    roi_head=dict(
    bbox_head=dict(
        # explicitly over-write all the `num_classes` field from default 15 to 5.
        num_classes=16)))

resume_from='/teamspace/studios/this_studio/mmrotate/demo/rotated_frcnn_4_lr/epoch_24 copy.pth'
evaluation = dict(interval=1, metric='mAP')
optimizer = dict(type='AdamW', lr=0.001, weight_decay=5e-05)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    policy='CosineAnnealing',
    min_lr=1e-06)