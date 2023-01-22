## noise-free degradations with isotropic Gaussian blurs
# training knowledge distillation
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main_iso_stage4.py --dir_data='/root/datasets' \
               --model='blindsr' \
               --scale='4' \
               --blur_type='iso_gaussian' \
               --noise=0 \
               --sig_min=0.2 \
               --sig_max=4.0 \
               --sig=3.6 \
               --n_GPUs=4 \
               --epochs_encoder 0 \
               --epochs_sr 600 \
               --lr_decay_sr 150 \
               --data_test Set14 \
               --st_save_epoch 0 \
               --data_train DF2K \
               --save 'KDSRsLx4_iso_stage4'\
               --pre_train_TA="experiment/KDSRsLx4_iso_stage3_x4_bicubic_iso/model/model_TA_last.pt" \
               --pre_train_ST="experiment/KDSRsLx4_iso_stage3_x4_bicubic_iso/model/model_TA_last.pt" \
               --lr_encoder 2e-4 \
               --patch_size 64 \
               --batch_size 64 \
               --resume 0 \
               --n_feats 128 \
               --n_blocks 28 \
               --n_resblocks 5


