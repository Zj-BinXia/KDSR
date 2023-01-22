CUDA_VISIBLE_DEVICES=4 python3 main_anisonoise_stage4.py --dir_data='/root/datasets' \
               --model='blindsr' \
               --scale='4' \
               --blur_type='aniso_gaussian' \
               --lambda_1 3.5 \
               --lambda_2 1.5 \
               --theta 30 \
               --n_GPUs=1 \
               --data_test Set14 \
               --save 'KDSRsM_anisonoise_test'\
               --pre_train_ST="./experiment/KDSRsM_anisonoise_x4.pt" \
               --resume 0 \
               --test_only \
               --save_results False

