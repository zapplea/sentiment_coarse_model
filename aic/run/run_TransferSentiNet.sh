#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="che313"
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --mem=10GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:3
echo "loading"
module load python/3.6.1
module load cudnn/v6
module load cuda/8.0.61
module load tensorflow/1.6.0-py36-gpu
echo "loaded"

python ../learn/LEARN_TransferSentiNet.py