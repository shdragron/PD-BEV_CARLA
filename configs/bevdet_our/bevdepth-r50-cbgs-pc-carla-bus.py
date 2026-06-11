# EVAL config for PD-BEV CARLA bus oracle (BEVDepth_DG inference; loads the DG ckpt).
_base_ = ['./bevdepth-r50-cbgs-pc-carla.py']
data = dict(
    val=dict(ann_file='data/bevdet_infos/bus_infos_val.pkl'),
    test=dict(ann_file='data/bevdet_infos/bus_infos_val.pkl'))
