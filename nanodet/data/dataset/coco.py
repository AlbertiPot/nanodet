import os
import torch
import numpy as np
import cv2
from pycocotools.coco import COCO
from .base import BaseDataset


class CocoDataset(BaseDataset):

    def get_data_info(self, ann_path):
        """
        Load basic information of dataset such as image path, label and so on.
        :param ann_path: coco json file path
        :return: image info:
        [{'license': 2,
          'file_name': '000000000139.jpg',
          'coco_url': 'http://images.cocodataset.org/val2017/000000000139.jpg',
          'height': 426,
          'width': 640,
          'date_captured': '2013-11-21 01:34:01',
          'flickr_url': 'http://farm9.staticflickr.com/8035/8024364858_9c41dc1666_z.jpg',
          'id': 139},
         ...
        ]
        """
        self.coco_api = COCO(ann_path)
        self.cat_ids = sorted(self.coco_api.getCatIds())                        # 获得全部数据集80个类category的id
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}   # 根据category_ids生成连续的标签，因为80个类的id存在间断（超类），存储格式是dict：{catid:label}，key是类id，value是label
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.img_ids = sorted(self.coco_api.imgs.keys())                        # 提取全部11万张图的id
        img_info = self.coco_api.loadImgs(self.img_ids)                         # 根据img id提取11万张图像的info
        return img_info

    # target: 提取单张img的info
    def get_per_img_info(self, idx):
        img_info = self.data_info[idx]
        file_name = img_info['file_name']
        height = img_info['height']
        width = img_info['width']
        id = img_info['id']
        if not isinstance(id, int):
            raise TypeError('Image id must be int.')
        info = {'file_name': file_name,
                'height': height,
                'width': width,
                'id': id}
        return info

    # target：提取单张图的标注和框，返回一个annotation包含bbox和gtlabels等
    def get_img_annotation(self, idx):
        """
        load per image annotation
        :param idx: index in dataloader
        :return: annotation dict
        """
        img_id = self.img_ids[idx]
        ann_ids = self.coco_api.getAnnIds([img_id])                             # 根据img_id获得annot_id，annot的存储格式：annotation{"id" : ~, "image_id" :~, "category_id" ;~ }
        anns = self.coco_api.loadAnns(ann_ids)                                  # 根据annot_id读取anns
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        if self.use_instance_mask:
            gt_masks = []
        if self.use_keypoint:
            gt_keypoints = []
        for ann in anns:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]                                     # x,y,w,h → x1,y1,x2,y2
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)                                   # 框中要是有多个目标，省略，anno中的iscrowd标签
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])            # 根据类id作为key，索引对应的label
                if self.use_instance_mask:
                    gt_masks.append(self.coco_api.annToMask(ann))
                if self.use_keypoint:
                    gt_keypoints.append(ann['keypoints'])
        if gt_bboxes:                                                           # 若gt_bboxes非空，转为numpy数组
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        annotation = dict(                                                      # 框和lable存为一个dict
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)
        if self.use_instance_mask:
            annotation['masks'] = gt_masks
        if self.use_keypoint:
            if gt_keypoints:
                annotation['keypoints'] = np.array(gt_keypoints, dtype=np.float32)
            else:
                annotation['keypoints'] = np.zeros((0, 51), dtype=np.float32)
        return annotation

    
    def get_train_data(self, idx):
        """
        Load image and annotation
        :param idx:
        :return: meta-data (a dict containing image, annotation and other information)
        """
        img_info = self.get_per_img_info(idx)
        file_name = img_info['file_name']
        image_path = os.path.join(self.img_path, file_name)                     # img的存储路径
        img = cv2.imread(image_path)                                            # 读图
        if img is None:
            print('image {} read failed.'.format(image_path))
            raise FileNotFoundError('Cant load image! Please check image path!')
        ann = self.get_img_annotation(idx)
        meta = dict(img=img,                                                    # 将图，info，anno中的box和label打包为一个字典
                    img_info=img_info,
                    gt_bboxes=ann['bboxes'],
                    gt_labels=ann['labels'])
        if self.use_instance_mask:
            meta['gt_masks'] = ann['masks']
        if self.use_keypoint:
            meta['gt_keypoints'] = ann['keypoints']

        meta = self.pipeline(meta, self.input_size)
        meta['img'] = torch.from_numpy(meta['img'].transpose(2, 0, 1))          # 交换RGB
        return meta

    def get_val_data(self, idx):
        """
        Currently no difference from get_train_data.
        Not support TTA(testing time augmentation) yet.
        :param idx:
        :return:
        """
        # TODO: support TTA
        return self.get_train_data(idx)
