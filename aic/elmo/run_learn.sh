#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="elmo train"
#SBATCH --time=140:00:00
#SBATCH --nodes=1
#SBATCH --mem=200GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
echo "loading"
module load python/3.6.1
module load tensorflow/1.6.0-py36-gpu
echo "loaded"
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