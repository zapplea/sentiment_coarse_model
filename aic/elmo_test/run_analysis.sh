#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="sentiment net"
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --mem=200GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
module load python/3.6.1

python data_analysis.py \
    --train_prefix='/datastore/liu121/sentidata2/data/bilm/training-monolingual.tokenized.shuffled/*' \
    --vocab_file '/datastore/liu121/sentidata2/data/bilm/vocab.txt' \
    --save_dir '/datastore/liu121/sentidata2/result/bilm/'