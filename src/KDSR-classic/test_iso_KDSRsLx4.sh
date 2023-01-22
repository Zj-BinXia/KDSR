CUDA_VISIBLE_DEVICES=4 python3 main_iso_stage4.py --test_only \
               --dir_data='/root/datasets' \
               --data_test='Set5+Set14+B100+Urban100+MANGA109' \
               --model='blindsr' \
               --scale='4' \
               --resume=0 \
               --pre_train_ST 'experiment/KDSRsL_iso_x4.pt' \
               --n_GPUs=1 \
               --save 'KDSRsL_iso_test'\
               --blur_type='iso_gaussian' \
               --noise 0 \
               --save_results False \
               --n_feats 128 \
               --n_blocks 28 \
               --n_resblocks 5
             