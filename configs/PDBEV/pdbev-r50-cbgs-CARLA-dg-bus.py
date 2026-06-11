# PD-BEV CARLA BUS oracle: identical recipe to the sedan run, only the per-vehicle
# info pkls (and thus the eval DB v1.0-carla_bus_eval) change.
_base_ = ['./pdbev-r50-cbgs-CARLA-dg.py']

data = dict(
    train=dict(ann_file='data/bevdet_infos/bus_infos_train.pkl'),
    val=dict(ann_file='data/bevdet_infos/bus_infos_val.pkl'),
    test=dict(ann_file='data/bevdet_infos/bus_infos_val.pkl'))

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(project='PDBEV-CARLA', entity='Robust_Ex', name='pdbev_carla_bus')),
    ])
