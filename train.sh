# use pytorch-lighting with distributed training
# nohup \
# /home/gbc/.conda/envs/nanodet/bin/python tools/train.py nanodet-m.yml \
# &
############ not convergence

# tools/deprecated/train.py with single gpu
# nohup \
# /home/gbc/.conda/envs/nanodet/bin/python \
# tools/deprecated/train.py \
# mytrain_coco_bs192_lr14_280eps_gpu1/nanodet-m.yml \
# &
# tensorboard --logdir=/home/gbc/workspace/nanodet/mytrain/coco_bs192_lr14_280eps_gpu1 --port=10001 --bind_all &
############ convergence mAP0.2047 vs 20.6 announced

# tools/deprecated/train.py with distributed training
nohup \
/home/gbc/.conda/envs/nanodet/bin/python \
-m torch.distributed.launch \
--nproc_per_node=2 \
--master_port 29501 \
tools/deprecated/train.py \
mytrain/coco_bs96mul2_lr14_280eps_gpu2_distri/nanodet-m.yml \
&
# nohup tensorboard --logdir=/home/gbc/workspace/nanodet/mytrain/coco_bs96mul2_lr14_280eps_gpu2_distri --port=10001 --bind_all &
############ convergence mAP0.2034 vs 20.6 announced