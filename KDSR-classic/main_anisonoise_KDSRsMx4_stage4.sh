## noise-free degradations with isotropic Gaussian blurs
# training knowledge distillation
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 main_anisonoise_stage4.py --dir_data='/root/datasets' \
               --model='blindsr' \
               --scale='4' \
               --blur_type='aniso_gaussian' \
               --noise=25.0 \
               --lambda_min=0.2 \
               --lambda_max=4.0 \
               --lambda_1 4.0 \
               --lambda_2 1.5 \
               --theta 120 \
               --n_GPUs=4 \
               --epochs_encoder 100 \
               --epochs_sr 500 \
               --data_test Set14 \
               --st_save_epoch 495 \
               --data_train DF2K \
               --save 'KDSRsMx4_anisonoise_stage4'\
               --pre_train_TA="./experiment/KDSRsMx4_anisonoise_stage3_x4_bicubic_aniso/model/model_TA_last.pt" \
               --pre_train_ST="./experiment/KDSRsMx4_anisonoise_stage3_x4_bicubic_aniso/model/model_TA_last.pt" \
               --resume 0 \
               --lr_encoder 2e-4 \
               --patch_size 64 \
               --batch_size 64 \


