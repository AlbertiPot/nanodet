import copy
from .coco import CocoDataset
from .xml_dataset import XMLDataset
from .wider_challenge import WiderChallengeDataset


def build_dataset(cfg, mode):
    dataset_cfg = copy.deepcopy(cfg)
    name = dataset_cfg.pop('name')
    if name == 'coco':
        return CocoDataset(mode=mode, **dataset_cfg)
    if name == 'wider_challenge':
        return WiderChallengeDataset(mode=mode, **dataset_cfg)
    if name == 'xml_dataset':
        return XMLDataset(mode=mode, **dataset_cfg)
    else:
        raise NotImplementedError('Unknown dataset type!')
