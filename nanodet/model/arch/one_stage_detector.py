import time
import torch
import torch.nn as nn
from ..backbone import build_backbone
from ..fpn import build_fpn
from ..head import build_head


class OneStageDetector(nn.Module):
    def __init__(self,
                 backbone_cfg,
                 fpn_cfg=None,
                 head_cfg=None,):
        super(OneStageDetector, self).__init__()
        self.backbone = build_backbone(backbone_cfg)            # 默认创建的shufflenetv2下载了pytorch的预训练模型
        if fpn_cfg is not None:
            self.fpn = build_fpn(fpn_cfg)
        if head_cfg is not None:
            self.head = build_head(head_cfg)

    def forward(self, x):
        x = self.backbone(x)
        if hasattr(self, 'fpn'):
            x = self.fpn(x)
        if hasattr(self, 'head'):
            x = self.head(x)
        return x

    def inference(self, meta):
        with torch.no_grad():
            torch.cuda.synchronize()
            time1 = time.time()
            preds = self(meta['img'])
            torch.cuda.synchronize()
            time2 = time.time()
            print('forward time: {:.3f}s'.format((time2 - time1)), end=' | ')
            results = self.head.post_process(preds, meta)
            torch.cuda.synchronize()
            print('decode time: {:.3f}s'.format((time.time() - time2)), end=' | ')
        return results

    def forward_train(self, gt_meta):
        preds = self(gt_meta['img'])                            # 等价于model(x), 其中gt_meta['img']是传入的ground truth数据，取其中img部分，过一遍model得到预测值
        loss, loss_states = self.head.loss(preds, gt_meta)      # 计算loss, 注意这里loss写在了head里

        return preds, loss, loss_states
