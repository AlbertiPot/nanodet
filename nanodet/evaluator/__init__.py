from .coco_detection import CocoDetectionEvaluator
from .wider_ped_detection import WiderPedDetectionEvaluator


def build_evaluator(cfg, dataset):
    if cfg.evaluator.name == 'CocoDetectionEvaluator':
        return CocoDetectionEvaluator(dataset)
    if cfg.evaluator.name == 'WiderPedDetectionEvaluator':
        return WiderPedDetectionEvaluator(dataset)
    else:
        raise NotImplementedError
