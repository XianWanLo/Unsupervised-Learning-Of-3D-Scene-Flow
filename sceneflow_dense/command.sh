#setup
"ops_pytorch ---> fused_conv_select
             ---> gpu_threenn_sample"

"Pointnet2"
python setup.py install 

#docker 
docker exec -it ljw_6003 bin/bash
watch -n 1 nvidia-smi

#train + evaluate
CUDA_VISIBLE_DEVICES=0 python train.py config.yaml      ## without log 
CUDA_VISIBLE_DEVICES=0 nohup python train_all.py config.yaml > log_0323_ChamferHalf.txt 2>&1 &      ## log

#evaluate only 
CUDA_VISIBLE_DEVICES=0 python evaluate.py config.yaml      ## without log 
CUDA_VISIBLE_DEVICES=1 nohup python evaluate.py config.yaml > log_12_21.txt 2>&1 &      ## log

