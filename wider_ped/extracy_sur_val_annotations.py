# target: 将原始的val.json中的sur图片和对应的注释提取出来，通过去掉ad图片和对应的annotation实现，存储为val_sur.json

import os
import json

if __name__ == '__main__':
    annFolder = '/home/gbc/workspace/nanodet/wider_ped'
    annName = 'val.json'
    annNewname = 'sur_val.json'
    annPath = os.path.join(annFolder, annName)
    annNewpath = os.path.join(annFolder, annNewname)
    
    print('loading annotations into memory...')
    with open(annPath, 'r') as f:
        annFile = json.load(f)
    assert type(annFile) == dict, 'annotation file format {} not supported'.format(type(annFile))
    originImglength = len(annFile["images"])
    originAnnlength = len(annFile["annotations"])

    adimgsCounts = 0
    imgIdlist = []                                  # ad图的主键
    imgDelindexlist = []                            # ad图在json list中的下标，根据下标删除list中的ad元素
    # 统计ad图的主键和下标
    for i, imgs in enumerate(annFile["images"]):
        if "ad" in imgs["file_name"]:
            adimgsCounts +=1
            imgIdlist.append(imgs["id"])
            imgDelindexlist.append(i)
    assert adimgsCounts == len(imgIdlist) == len(imgDelindexlist),'the counts of ad imgs are not correct, please check'

    # 删除annFile["images"]不是sur图片的小字典，动态删除list参考https://blog.csdn.net/exm_further/article/details/112251558?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_baidulandingword-0&spm=1001.2101.3001.4242
    for counter, index in enumerate(imgDelindexlist):
        index = index - counter
        annFile["images"].pop(index)
    assert len(annFile["images"]) ==  originImglength - adimgsCounts, 'delete wrong number of ad images'
    
    adannsCounts = 0                                # ad标注的主键
    adannsList = []                                 # ad标注的下标，根据此下标删除list中ad标注
    for i, anns in enumerate(annFile['annotations']):
        if anns['image_id'] in imgIdlist:           # 标注对应的图是ad图
            adannsCounts += 1
            adannsList.append(i)
        else:                                       # 不是ad图，计算其area
            anns['area'] = anns['bbox'][2] * anns['bbox'][3]
            #import pdb;pdb.set_trace()
    assert adannsCounts == len(adannsList),'the counts of ad annotation are not correct'
    
    # 删除annFile["annotations"]不是sur的标注的小字典
    for counter, index in enumerate(adannsList):
        index = index - counter
        annFile["annotations"].pop(index)
    assert len(annFile["annotations"]) == originAnnlength - adannsCounts, 'delete wrong number of ad annotations'

    
    with open(annNewpath, 'w') as f:
        json.dump(annFile, f)
    
    print("finished extractoin")
    