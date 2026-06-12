# PD-BEV CARLA BUS oracle — attempt-5 recipe (OPTION-2: authors-native depth regime,
# from scratch). Same rationale as the suv config (see there): native input 384x704 +
# native depth grid [1,100,1] to widen the sparse-depth-supervision anchoring margin.
# Bus is the tallest platform (cams 2.87-4.08m, pitch +20deg) — most collapse-prone.
_base_ = ['./pdbev-r50-cbgs-CARLA-dg.py']

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian']

data_config = {
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (384, 704),
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
    'depth': [1.0, 100.0, 1.0],
}

model = dict(
    img_view_transformer=dict(
        grid_config=grid_config,
        input_size=data_config['input_size']))

train_pipeline = [
    dict(type='PrepareImageInputs_UDA', is_train=True, data_config=data_config),
    dict(type='LoadAnnotationsBEVDepth', bda_aug_conf={{_base_.bda_aug_conf}},
         classes=class_names),
    dict(type='CarlaDPTMultiViewDepth_UDA', downsample=1, grid_config=grid_config),
    dict(type='Load3DBoxesHeatmap', classes=class_names, downsample_feature=8),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_depth', 'gt_depth_real',
                                 'heatmaps_2d', 'ann_maps_2d', 'heatmap_masks_2d',
                                 'heatmaps_2d_aug', 'ann_maps_2d_aug', 'heatmap_masks_2d_aug', 'bev_aug'])
]

optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=2,
                        grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='step', warmup='linear', warmup_iters=1000, warmup_ratio=0.001,
                 step=[19, 23])

data = dict(
    samples_per_gpu=16,
    train=dict(ann_file='data/bevdet_infos/bus_infos_train.pkl',
               pipeline=train_pipeline),
    val=dict(ann_file='data/bevdet_infos/bus_infos_val.pkl'),
    test=dict(ann_file='data/bevdet_infos/bus_infos_val.pkl'))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='PDBEV-CARLA', entity='Robust_Ex', name='pdbev_carla_bus_native384')),
    ])
