export CUDA_VISIBLE_DEVICES=4
python -m torch.distributed.launch \
--nproc_per_node=1 train2.py \
--respath checkpoints/train_STDC1-Seg/ \
--backbone STDCNet813 \
--mode trainval \
--n_workers_train 12 \
--n_workers_val 1 \
--max_iter 120000 \
--use_boundary_2 False \
--use_boundary_8 True \
--use_boundary_4 False \
--pretrain_path checkpoints/STDCNet813M_73.91.tar
