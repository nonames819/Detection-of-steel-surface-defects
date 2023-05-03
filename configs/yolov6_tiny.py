# YOLOv6t model
model = dict(
    type='YOLOv6t',
    # pretrained="../last_ckpt.pt",
    pretrained=None,
    depth_multiple=0.40,
    width_multiple=0.50,
    backbone=dict(
        type='EfficientRep',
        num_repeats=[1, 6, 12, 18, 6],
        out_channels=[64, 128, 256, 512, 1024],
        ),
    neck=dict(
        type='RepPAN',
        num_repeats=[12, 12, 12, 12],
        out_channels=[256, 128, 128, 256, 256, 512],
        ),
    head=dict(
        type='EffiDeHead',
        in_channels=[128, 256, 512],
        num_layers=3,
        anchors=1,
        strides=[8, 16, 32],
        iou_type='siou'
    )
)

solver = dict(
    optim='SGD',
    lr_scheduler='Cosine',
    lr0=0.03,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1
)

data_aug = dict(
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    degrees=90.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    flipud=0.5,
    fliplr=0.5,
    mosaic=0.0,
    mixup=1.0,
)