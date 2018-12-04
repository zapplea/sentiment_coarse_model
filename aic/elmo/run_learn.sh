#!/bin/bash

python train_elmo.py \
    --train_prefix='/datastore/liu121/sentidata2/data/bilm/training-monolingual.tokenized.shuffled/*' \
    --vocab_file '/datastore/liu121/sentidata2/data/bilm/vocab.txt' \
    --save_dir '/datastore/liu121/sentidata2/result/bilm/'