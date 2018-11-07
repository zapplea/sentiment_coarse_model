#!/bin/bash

python LEARN_elmo.py \
    --train_file='/datastore/liu121/sentidata2/data/aic2018/coarse_data/train_coarse.pkl' \
    --save_dir '/datastore/liu121/sentidata2/result/bilm/' \
    --gpu_num 1