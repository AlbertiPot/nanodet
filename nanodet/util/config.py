from .yacs import CfgNode

# 整个yaml文件存储为一个CfgNode对象（树状的字典）：最底层key赋值，中间key仍要创建一个子CfgNode。最后的字典是大CfgNode嵌套多个小CfgNode
# myfork from rgbgirshick : https://github.com/AlbertiPot/yacs.git
cfg = CfgNode(new_allowed=True)                         # 总yaml文件对应的CfgNode
cfg.save_dir = './'                                     # save_dir的默认值，后期会被merge掉
# common params for NETWORK
cfg.model = CfgNode()                                   # 第一个子CfgNode，存储模型的信息
cfg.model.arch = CfgNode(new_allowed=True)
cfg.model.arch.backbone = CfgNode(new_allowed=True)
cfg.model.arch.neck = CfgNode(new_allowed=True)
cfg.model.arch.head = CfgNode(new_allowed=True)

# DATASET related params                                # 第二个子CfgNode，存储数据集信息
cfg.data = CfgNode(new_allowed=True)
cfg.data.train = CfgNode(new_allowed=True)
cfg.data.val = CfgNode(new_allowed=True)
cfg.device = CfgNode(new_allowed=True)
# train
cfg.schedule = CfgNode(new_allowed=True)

# logger
cfg.log = CfgNode()
cfg.log.interval = 50

# testing
cfg.test = CfgNode()
# size of images for each device


def load_config(cfg, args_cfg):
    cfg.defrost()                                   # 允许修改cfg
    cfg.merge_from_file(args_cfg)                   # 合并
    cfg.freeze()                                    # 不允许修改


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as f:
        print(cfg, file=f)
