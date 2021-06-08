# nohup \
# CUDA_VISIBLE_DEVICES=1 \
/home/gbc/.conda/envs/nanodet/bin/python \
/home/gbc/workspace/nanodet/tools/deprecated/train.py \
nanodet-m.yml \
# &

# nohup tensorboard --logdir=/home/gbc/workspace/nanodet/wider_ped/test --port=9998 --bind_all &