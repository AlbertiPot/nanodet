CUDA_VISIBLE_DEVICES=0 \
nohup \
/home/gbc/.conda/envs/nanodet/bin/python \
/home/gbc/workspace/nanodet/tools/deprecated/train.py \
nanodet-m.yml \
&

# nohup tensorboard --logdir=/home/gbc/workspace/nanodet/wider_ped/  --port=10000 --bind_all &