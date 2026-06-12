# PD-BEV CARLA SEDAN retrain — attempt-5 (OPTION-2: authors-native depth regime, from scratch).
#
# Root cause of attempts 1/2/4 (probed): PD-BEV supervises depth ONLY at GT box centers
# (~5% of pixels, virtual depth = real*450/fx'); the other 95% are anchored solely by the
# detection gradient through the LSS lift. Under suv geometry (cams +0.75m, content farther/
# sparser) the from-scratch depthnet fell into a near-bin delta attractor (softmax collapsed
# to bin 1/112, logits ~9k) -> all features splat into an ego-ring -> degenerate detection
# (NDS 0.0). Our depth grid [2,58,0.5] (BEVDepth-matched) also truncated ~40% of the
# virtual-depth targets vs the authors' native grid, shaving the anchoring margin.
#
# OPTION-2 restores the AUTHORS' native depth regime to widen that margin:
#   * input_size (384,704)  -- native vertical FOV (crop_top ~12px, not 140)
#   * depth grid [1,100,1]  -- native virtual-depth bin range (in-bin ~88% vs ~60%)
#   * from scratch, native loss_depth_weight 3.0; stabilizers kept (clip 35, warmup 1000)
#   * DPT dense gt_depth kept in the pipeline (vestigial for the loss; ann_maps drive it)
_base_ = ['./pdbev-r50-cbgs-CARLA-dg.py']

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian']

data_config = {
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (384, 704),          # authors-native (was 256x704)
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
    'depth': [1.0, 100.0, 1.0],        # authors-native (was [2,58,0.5])
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
    samples_per_gpu=16,               # keep eff-batch 64 (16 x 2gpu x accum2); ~75GB/GPU at 384x704
    train=dict(ann_file='data/bevdet_infos/sedan_infos_train.pkl',
               pipeline=train_pipeline),
    val=dict(ann_file='data/bevdet_infos/sedan_infos_val.pkl'),
    test=dict(ann_file='data/bevdet_infos/sedan_infos_val.pkl'))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='PDBEV-CARLA', entity='Robust_Ex', name='pdbev_carla_sedan_native384')),
    ])
