_base_ = '../configs/fastscnn/fast_scnn_8xb4-160k_cityscapes-512x1024.py'

# data setting
data_root = './datasets/thyroid_seg/'
metainfo = {
    'classes': ('background', 'roi',),
    'palette': [
        (127, 127, 127),
        (220, 20, 60),
    ]
}
num_classes=2
crop_size = (
    512,
    512,
)

# model setting
model = dict(
    data_preprocessor=dict(size=crop_size),
    test_cfg=dict(mode='whole', crop_size=crop_size[0]),
    decode_head=dict(num_classes=num_classes)
)

# training config
train_cfg = dict(max_iters=20000, type='IterBasedTrainLoop', val_interval=500)

# batch size and learning rate
train_batch_size_per_gpu = 32
val_batch_size_per_gpu = 1
test_batch_size_per_gpu = 1
num_gpu = 2
adjust_factor = 1 
base_lr = adjust_factor * num_gpu * train_batch_size_per_gpu * 0.1 / (4*8)

# default setting
train_num_workers = 8
val_num_workers = 8
test_num_workers = 8
optim_wrapper = dict(optimizer=dict(lr=base_lr))
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=5000, max_keep_ckpts=2, save_best='mIoU'),
    logger=dict(log_metric_by_epoch=False, interval=1000, type='LoggerHook'),
)


# pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(2048, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
test_pipeline = val_pipeline
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

# dataloader
train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    dataset=dict(
        data_prefix=dict(img_path='train/images', seg_map_path='train/masks'),
        data_root=data_root,
        pipeline=train_pipeline,
        type='myDataset'),
    num_workers=train_num_workers)
val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    dataset=dict(
        data_prefix=dict(img_path='val/images', seg_map_path='val/masks'),
        data_root=data_root,
        pipeline=val_pipeline,
        type='myDataset'),
    num_workers=val_num_workers)
test_dataloader = dict(
    batch_size=test_batch_size_per_gpu,
    dataset=dict(
        data_prefix=dict(img_path='test/images', seg_map_path='test/masks'),
        data_root=data_root,
        pipeline=val_pipeline,
        type='myDataset'),
    num_workers=test_num_workers)

# evaluator
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
        'mDice',
        'mFscore',
    ], type='IoUMetric')
test_evaluator = val_evaluator



