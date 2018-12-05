#!/bin/bash

python LEARN_elmo.py \
    --train_prefix='/datastore/liu121/sentidata2/data/bilm/aic2018/*' \
    --vocab_file '/datastore/liu121/sentidata2/data/bilm/vocab_aic.txt' \
    --save_dir '/datastore/liu121/sentidata2/result/bilm/'