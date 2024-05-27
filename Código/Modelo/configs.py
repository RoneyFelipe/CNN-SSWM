#%%writefile /content/mmpretrain/configs/mobilenet_v2/mobilenet_v2_1x_coins.py
%%writefile /home/anderson/mmsegmentation// --definir modelo
_base_ = [
    '../_base_/models/mobilenet_v2_1x.py',
    '../_base_/schedules/imagenet_bs256_epochstep.py',
    '../_base_/default_runtime.py'
]

# ---- configurações do modelo ----

model = dict(
    backbone=dict(
        init_cfg = dict(
            type='Pretrained', #Mudar aquiiiii 
            checkpoint='https://download.openmmlab.com/mmclassification/v0/mobilenet_v2/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth',
            prefix='backbone')
    ),
    head=dict(num_classes=5))

# ---- data settings ----

dataset_type = 'CustomDataset'
data_preprocessor = dict(
    mean=[124.508, 116.050, 106.438],
    std=[58.577, 57.310, 57.437],
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=64, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=72, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=64),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='/content/drive/MyDrive/coins_dataset/split/train',
        classes=['colonhao', 'folhaLarga', 'graminea', 'mamona'],
        pipeline=train_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='/content/drive/MyDrive/coins_dataset/split/val',
        classes=['colonhao', 'folhaLarga', 'graminea', 'mamona'],
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

test_dataloader = dict(
    batch_size=64,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        data_prefix='/content/drive/MyDrive/coins_dataset/split/test',
        classes=['colonhao', 'folhaLarga', 'graminea', 'mamona'],
        pipeline=test_pipeline,
    ),
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# Especificar a metrica
val_evaluator = dict(type='Accuracy', topk=1)
test_evaluator = val_evaluator

# ---- schedule ----
optim_wrapper=dict(
    optimizer=dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001),
    clip_grad=None,
)
# learning rate scheduler
param_scheduler = dict(type='StepLR', by_epoch=True, step_size=1, gamma=0.1)

# Numero de epocas e intervalo de validacao
train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
# Use configs default
val_cfg = dict()
test_cfg = dict()

# ---- runtime ----

default_hooks = dict(logger=dict(interval=10))

randomness = dict(seed=0, deterministic=False)