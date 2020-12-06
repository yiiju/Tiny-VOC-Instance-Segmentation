python3 test.py --test True \
                --gpuid 5 \
                --batch_size 1 \
                --save_dir ./test/sgd_pretrain_val20 \
                --reset True \
                --checkpoint ./train/sgd_pretrain_val20/model/model_00080.pth \
                --outjson epoch80.json \
                --log_file_name test.log \
                --mask_threshold 0.45 \
                --norm False