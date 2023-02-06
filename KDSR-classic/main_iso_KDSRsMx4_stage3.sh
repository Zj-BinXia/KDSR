## noise-free degradations with isotropic Gaussian blurs
# training knowledge distillation
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 main_iso_stage3.py --dir_data='/root/datasets' \
               --model='blindsr' \
               --scale='4' \
               --blur_type='iso_gaussian' \
               --noise=0 \
               --sig_min=0.2 \
               --sig_max=4.0 \
               --sig=3.6 \
               --n_GPUs=4 \
               --epochs_encoder 0 \
               --epochs_sr 500 \
               --data_test Set14 \
               --st_save_epoch 495 \
               --data_train DF2K \
               --save 'KDSRsMx4_iso_stage3'\
               --patch_size 64 \
               --batch_size 64 \


