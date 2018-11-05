#!/bin/bash

python bin/run_test.py \
    --train_prefix='/datastore/liu121/sentidata2/data/bilm/training-monolingual.tokenized.shuffled/news.en.heldout-000*' \
    --vocab_file '/datastore/liu121/sentidata2/data/bilm/vocab.txt' \
    --save_dir '/datastore/liu121/sentidata2/result/bilm/'