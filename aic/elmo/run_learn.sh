#!/bin/bash

if [ $1 = 'cn' ]
then
    python LEARN_elmo.py \
        --train_prefix='/datastore/liu121/sentidata2/data/bilm/aic2018/*' \
        --vocab_file '/datastore/liu121/sentidata2/data/bilm/vocab_aic.txt' \
        --save_dir '/datastore/liu121/sentidata2/result/bilm/'
elif [ $1 = 'en' ]
then
    python LEARN_elmo.py \
        --train_prefix='/datastore/liu121/sentidata2/data/bilm/training-monolingual.tokenized.shuffled/*' \
        --vocab_file '/datastore/liu121/sentidata2/data/bilm/vocab.txt' \
        --save_dir '/datastore/liu121/sentidata2/result/bilm/'
fi