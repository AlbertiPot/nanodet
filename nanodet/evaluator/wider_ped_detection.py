import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import json
import os
import copy


def xyxy2xywh(bbox):
    """
    change bbox to coco format
    :param bbox: [x1, y1, x2, y2]
    :return: [x, y, w, h]
    """
    return [
        bbox[0],
        bbox[1],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    ]


class WiderPedDetectionEvaluator:
    def __init__(self, dataset):
        assert hasattr(dataset, 'coco_api')
        self.coco_api = dataset.coco_api
        self.cat_ids = dataset.cat_ids
        self.metric_names = ['mAP', 'AP_50', 'AP_75', 'AP_small', 'AP_m', 'AP_l']

    def results2json(self, results):
        """
        results: {image_id: {label: [bboxes...] } }
        :return coco json format: {image_id:
                                   category_id:
                                   bbox:
                                   score: }
        """
        json_results = []
        for image_id, dets in results.items():
            for label, bboxes in dets.items():
                category_id = self.cat_ids[label]
                for bbox in bboxes:
                    score = float(bbox[4])
                    detection = dict(
                        image_id=int(image_id),
                        category_id=int(category_id),
                        bbox=xyxy2xywh(bbox),
                        score=score)
                    json_results.append(detection)
        return json_results

    def evaluate(self, results, save_dir, rank=-1):
        results_json = self.results2json(results)
        json_path = os.path.join(save_dir, 'results{}.json'.format(rank))
        json.dump(results_json, open(json_path, 'w'))                                               # reults_json存入json_path打开的文件
        coco_dets = self.coco_api.loadRes(json_path)                                                # 读取刚保存的json结果
        coco_eval = COCOeval(copy.deepcopy(self.coco_api), copy.deepcopy(coco_dets), "bbox")        # 调用COCOeval实例化评估对象：传入dataset的coco_api作为ground_truth, 刚保存的json结果作为dets
        coco_eval.evaluate()                                                                        # 每张图eval
        coco_eval.accumulate()                                                                      # 积累图的结果
        coco_eval.summarize()                                                                       # 计算metrcis
        aps = coco_eval.stats[:6]                                                                   # 6个AP值
        eval_results = {}
        for k, v in zip(self.metric_names, aps):
            eval_results[k] = v
        return eval_results
