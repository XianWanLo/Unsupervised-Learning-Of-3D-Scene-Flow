
CUDA_VISIBLE_DEVICES=0 python evaluate.py config.yaml

CUDA_VISIBLE_DEVICES=0 nohup python train_color_no_depth.py config.yaml > log_train_color_no_depth.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 python train_color.py config.yaml