# EVAL config for the PD-BEV CARLA SEDAN (native384 retrain) (attempt-5, authors-native depth regime).
# MUST architecturally match the sedan384 training config: input 384x704, depth grid [1,100,1]
# (depthnet D=100), else the checkpoint does not load. Deterministic IDA, no aug.
_base_ = ['./bevdepth-r50-cbgs-pc-carla.py']

class_names = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian']

data_config = {
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (384, 704),
    'src_size': (900, 1600),
    'resize': (0.0, 0.0), 'rot': (0.0, 0.0), 'flip': False, 'crop_h': (0.0, 0.0),
    'resize_test': 0.0, 're_ratio': False,
    'pitch_aug': (0.0, 0.0), 'yaw_aug': (0.0, 0.0), 'roll_aug': (0.0, 0.0),
    'extri_x_aug': (0.0, 0.0), 'extri_y_aug': (0.0, 0.0), 'extri_z_aug': (0.0, 0.0),
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

bda_aug_conf = dict(rot_lim=(0.0, 0.0), scale_lim=(1.0, 1.0), flip_dx_ratio=0.0, flip_dy_ratio=0.0)
file_client_args = dict(backend='disk')

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

data = dict(
    val=dict(ann_file='data/bevdet_infos/sedan_infos_val.pkl', pipeline=test_pipeline),
    test=dict(ann_file='data/bevdet_infos/sedan_infos_val.pkl', pipeline=test_pipeline))
