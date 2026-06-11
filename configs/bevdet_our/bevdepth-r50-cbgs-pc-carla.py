# EVAL config for PD-BEV CARLA: plain BEVDepth_DG (working simple_test) that LOADS the
# PD-BEV (PCBEV_DG) trained checkpoint -- the DG aux heads (img_aug/bev_img_aux) are
# train-only and dropped on load (strict=False). Detection submodules (backbone/neck/
# view_transformer depthnet/bev_encoder/pts_bbox_head) match the training config exactly
# (grid depth[2,58,0.5]=112, input 256x704, 5 CARLA tasks) so the weights transfer.
# Use with pdbev_dump_val.py to dump val detections, then score with the verified CARLA NDS.
_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian']
class_names_train = class_names

data_config = {
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    'resize': (0.0, 0.0), 'rot': (0.0, 0.0), 'flip': False, 'crop_h': (0.0, 0.0),
    'resize_test': 0.0, 're_ratio': False,
    'pitch_aug': (0.0, 0.0), 'yaw_aug': (0.0, 0.0), 'roll_aug': (0.0, 0.0),
    'extri_x_aug': (0.0, 0.0), 'extri_y_aug': (0.0, 0.0), 'extri_z_aug': (0.0, 0.0),
}

grid_config = {
    'x': [-51.2, 51.2, 0.8], 'y': [-51.2, 51.2, 0.8], 'z': [-5, 3, 8],
    'depth': [2.0, 58.0, 0.5],   # 112 bins -- MUST match training (depthnet output dim)
}
voxel_size = [0.1, 0.1, 0.2]
numC_Trans = 80

model = dict(
    type='BEVDepth_DG',
    img_backbone=dict(
        pretrained='torchvision://resnet50', type='ResNet', depth=50, num_stages=4,
        out_indices=(2, 3), frozen_stages=-1, norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False, with_cp=True, style='pytorch'),
    img_neck=dict(type='CustomFPN', in_channels=[1024, 2048], out_channels=512,
                  num_outs=1, start_level=0, out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformerBEVDepth_DG', grid_config=grid_config,
        input_size=data_config['input_size'], in_channels=512, out_channels=numC_Trans,
        depthnet_cfg=dict(use_dcn=False), downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet', numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS', in_channels=numC_Trans * 8 + numC_Trans * 2, out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead', in_channels=256,
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
    train_cfg=dict(pts=dict(
        point_cloud_range=point_cloud_range, grid_size=[1024, 1024, 40],
        voxel_size=voxel_size, out_size_factor=8, dense_reg=1, gaussian_overlap=0.1,
        max_objs=500, min_radius=2,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    test_cfg=dict(pts=dict(
        pc_range=point_cloud_range[:2],
        post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        max_per_img=500, max_pool_nms=False,
        min_radius=[4, 12, 10, 1, 0.175], score_threshold=0.1,
        out_size_factor=8, voxel_size=voxel_size[:2], pre_max_size=1000, post_max_size=83,
        nms_type=['rotate', 'rotate', 'rotate', 'rotate', 'rotate'],
        nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2],
        nms_rescale_factor=[1.0, 0.7, 0.4, [1.0, 1.0], 1.5])))

dataset_type = 'CarlaPDBEVDataset'
data_root = 'data/nuscenes/'
infos_root = 'data/bevdet_infos/'
file_client_args = dict(backend='disk')
bda_aug_conf = dict(rot_lim=(0.0, 0.0), scale_lim=(1.0, 1.0), flip_dx_ratio=0.0, flip_dy_ratio=0.0)

test_pipeline = [
    dict(type='PrepareImageInputs_UDA', data_config=data_config),
    dict(type='LoadAnnotationsBEVDepth', bda_aug_conf=bda_aug_conf, classes=class_names, is_train=False),
    dict(type='LoadPointsFromFile_UDA', coord_type='LIDAR', load_dim=3, use_dim=3,
         file_client_args=file_client_args),
    dict(type='MultiScaleFlipAug3D', img_scale=(1600, 900), pts_scale_ratio=1, flip=False,
         transforms=[
             dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
             dict(type='Collect3D', keys=['points', 'img_inputs'])
         ])
]
input_modality = dict(use_lidar=False, use_camera=True, use_radar=False, use_map=False, use_external=False)
share_data_config = dict(type=dataset_type, classes=class_names_train, modality=input_modality)
test_data_config = dict(pipeline=test_pipeline, ann_file=infos_root + 'sedan_infos_val.pkl',
                        data_root=data_root)
data = dict(samples_per_gpu=8, workers_per_gpu=6, val=test_data_config, test=test_data_config)
for key in ['val', 'test']:
    data[key].update(share_data_config)
