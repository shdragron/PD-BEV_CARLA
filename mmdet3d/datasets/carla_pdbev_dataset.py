"""CARLA geobev adapter for PD-BEV (Generalizable-BEV).

Reuses the already-coordinate-VERIFIED BEVDet geobev pkls
(data/bevdet_infos/{sedan,suv,bus}_infos_{train,val}.pkl). `get_data_info` is
inherited from DeepAccidentDataset: it passes ``curr=info`` untouched, so all
camera geometry (lidar2img / sensor2ego / multi-view depth) is reconstructed
downstream by PrepareImageInputs_UDA / PointToMultiViewDepth_UDA from
sensor2ego_rotation/translation + ego2global + cam_intrinsic. Those were verified
byte-identical (lidar2cam max diff 4.66e-14) to the sensor2lidar reference and
projrate 1.0 against GT-on-vehicles. This is the quaternion path -- do NOT use the
built-in CarlaDataset (CARLA_dataset.py), which is SHIFT-flavored (scalar-pitch
rotation + hardcoded paths).
"""
import mmcv
import numpy as np
from mmdet.datasets import DATASETS

from .DeepAccident_dataset import DeepAccidentDataset


@DATASETS.register_module()
class CarlaPDBEVDataset(DeepAccidentDataset):
    CLASSES = ('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian')

    def load_annotations(self, ann_file):
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = dict(image_size=[900, 1600],
                             categories=list(self.CLASSES))
        self.version = 'v1.0-carla'
        return data_infos

    def get_cat_ids(self, idx):
        # geobev pkls carry no top-level 'gt_names'; derive class ids from
        # ann_infos[1] (label indices into CLASSES). Only invoked under CBGS.
        labels = np.asarray(self.data_infos[idx]['ann_infos'][1]).astype(int)
        return list({int(l) for l in labels if 0 <= int(l) < len(self.CLASSES)})
