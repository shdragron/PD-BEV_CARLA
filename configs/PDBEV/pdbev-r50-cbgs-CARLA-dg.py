# PD-BEV (Generalizable-BEV) on CARLA geobev sedan -- OPTION B (method as intended).
# DATA/SCHEDULE/HEAD matched to the BEVDepth-CARLA fair-comparison recipe
# (bevdepth/exps/nuscenes/carla/carla_base.py): input 256x704, BEV 128x128, depth
# [2,58,0.5]=112 bins, 6-class -> 5 CenterHead tasks, fp32, CBGS/EMA off,
# 24ep MultiStepLR[19,23], eff-batch 64 (16x2gpu x accum2), lr 2e-4.
# AUG: PD-BEV's NATIVE DG augmentation kept ON (extrinsic pitch/yaw/roll + IDA + BDA) -- this is
# the perspective-debiasing method; turning it off would neuter it. (The other detectors run
# no-aug; PD-BEV's aug IS its contribution -- report with the train-test-overlap caveat above.)
# Model = PD-BEV's PCBEV_DG (perspective-debiasing on BEVDepth).
# Residual diff vs BEVDepth: depth GT = PD-BEV lidar-projected (PointToMultiViewDepth_UDA) vs
#   BEVDepth DPT-dense -- d_bound range matched, source differs.
# Data = the coordinate-VERIFIED BEVDet geobev pkls (reused via symlink).
_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names_train = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian']
class_names_test = class_names_train
class_names = class_names_train

# Option B: PD-BEV's NATIVE DG augmentation kept ON (extrinsic pitch/yaw/roll + IDA + BDA)
# -- this is the method (perspective-debiasing via the augmented-extrinsic branch). Everything
# ELSE matched to BEVDepth-CARLA. input_size 256x704 (final crop), resize base 0.44.
# NOTE for VP interpretation: extrinsic yaw_aug +/-0.4rad(~23deg) overlaps the VP yaw test range;
# pitch/roll_aug ~+/-2.3deg are smaller than the VP test (up to +/-20deg).
data_config = {
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.1),
    'resize_test': 0.00,
    're_ratio': False,
    'pitch_aug': (-0.04, 0.04),
    'yaw_aug': (-0.4, 0.4),
    'roll_aug': (-0.04, 0.04),
    'extri_x_aug': (-2.0, 2.0),
    'extri_y_aug': (-2.0, 2.0),
    'extri_z_aug': (-2.0, 2.0),
}

grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [2.0, 58.0, 0.5],   # 112 bins, matches BEVDepth-CARLA d_bound
}

voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 80

model = dict(
    type='PCBEV_DG',
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet', depth=50, num_stages=4, out_indices=(2, 3),
        frozen_stages=-1, norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False, with_cp=True, style='pytorch'),
    img_aug=dict(
        type='Img_Aux', numC_input=512, class_name=class_names_train,
        upsample=[1, 2, 1],
        num_channels=[256, 128, len(class_names_train) + len(class_names_train) * 6]),
    bev_img_aux=dict(
        type='Img_Aux_Dy', numC_input=80, class_name=class_names_train,
        num_layer=[2, 1, 1], upsample=[2, 1, 1],
        num_channels=[256, 128, len(class_names_train) + len(class_names_train) * 6]),
    img_neck=dict(
        type='CustomFPN', in_channels=[1024, 2048], out_channels=512,
        num_outs=1, start_level=0, out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformer_pc',
        BEV_Aux=dict(
            type='BEV_Aux', class_name=class_names_train, numC_input=256,
            num_layer=[2, 2, 2], height_num=4,
            num_channels=[numC_Trans, numC_Trans, numC_Trans + 4]),
        downsample_from_ann=2, grid_config=grid_config,
        input_size=data_config['input_size'], in_channels=512,
        depthnet_cfg=dict(use_dcn=False), out_channels=numC_Trans,
        loss_depth_weight=3.0, downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet', numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS', in_channels=numC_Trans * 8 + numC_Trans * 2, out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead', in_channels=256,
        # 5 tasks matching BEVDepth-CARLA (task 0 = single class for PD-BEV bev_loss).
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=1, class_names=['truck']),
            dict(num_class=1, class_names=['bus']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=1, class_names=['pedestrian']),
        ],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder', pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500, score_threshold=0.1, out_size_factor=8,
            voxel_size=voxel_size[:2], code_size=9),
        separate_head=dict(type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range, grid_size=[1024, 1024, 40],
            voxel_size=voxel_size, out_size_factor=8, dense_reg=1,
            gaussian_overlap=0.1, max_objs=500, min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500, max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.175], score_threshold=0.1,
            out_size_factor=8, voxel_size=voxel_size[:2],
            pre_max_size=1000, post_max_size=83,
            nms_type=['rotate', 'rotate', 'rotate', 'rotate', 'rotate'],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2],
            nms_rescale_factor=[1.0, 0.7, 0.4, [1.0, 1.0], 1.5])))

# Data
dataset_type = 'CarlaPDBEVDataset'
data_root = 'data/nuscenes/'
infos_root = 'data/bevdet_infos/'
file_client_args = dict(backend='disk')

# BDA ON (PD-BEV native) -- part of the DG augmentation.
bda_aug_conf = dict(rot_lim=(-22.5, 22.5), scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5, flip_dy_ratio=0.5)

train_pipeline = [
    dict(type='PrepareImageInputs_UDA', is_train=True, data_config=data_config),
    dict(type='LoadAnnotationsBEVDepth', bda_aug_conf=bda_aug_conf, classes=class_names),
    dict(type='LoadPointsFromFile_UDA', coord_type='LIDAR', load_dim=3, use_dim=3,
         file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth_UDA', downsample=1, grid_config=grid_config),
    dict(type='Load3DBoxesHeatmap', classes=class_names_train, downsample_feature=8),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_depth', 'gt_depth_real',
                                 'heatmaps_2d', 'ann_maps_2d', 'heatmap_masks_2d',
                                 'heatmaps_2d_aug', 'ann_maps_2d_aug', 'heatmap_masks_2d_aug', 'bev_aug'])
]

test_pipeline = [
    dict(type='PrepareImageInputs_UDA', data_config=data_config),
    dict(type='LoadAnnotationsBEVDepth', bda_aug_conf=bda_aug_conf, classes=class_names_test, is_train=False),
    dict(type='MultiScaleFlipAug3D', img_scale=(1600, 900), pts_scale_ratio=1, flip=False,
         transforms=[
             dict(type='DefaultFormatBundle3D', class_names=class_names_test, with_label=False),
             dict(type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
         ])
]

input_modality = dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False)
share_data_config = dict(type=dataset_type, classes=class_names_train, modality=input_modality)
test_data_config = dict(pipeline=test_pipeline, ann_file=infos_root + 'sedan_infos_val.pkl')

data = dict(
    samples_per_gpu=16,          # 16 x 2 GPU x accum2 = eff-batch 64 (BEVDepth-CARLA)
    workers_per_gpu=6,
    train=dict(
        data_root=data_root, ann_file=infos_root + 'sedan_infos_train.pkl',
        pipeline=train_pipeline, classes=class_names_train,
        test_mode=False, box_type_3d='LiDAR'),
    val=test_data_config, test=test_data_config)
for key in ['val', 'test']:
    data[key].update(share_data_config)
data['train'].update(share_data_config)

# Optimizer: AdamW lr 2e-4, eff-batch 64 via grad accumulation (cumulative_iters=2), grad_clip 5.
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2)
optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=2,
                        grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(policy='step', warmup='linear', warmup_iters=200, warmup_ratio=0.001, step=[19, 23])
runner = dict(type='EpochBasedRunner', max_epochs=24)

# CBGS OFF, EMA OFF (fair comparison). NDS eval deferred to the model-agnostic benchmark.
evaluation = dict(interval=100)
custom_hooks = []
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='PDBEV-CARLA', entity='Robust_Ex', name='pdbev_carla_sedan')),
    ])
