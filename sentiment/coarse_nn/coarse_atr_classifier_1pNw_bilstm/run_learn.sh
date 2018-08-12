#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="transfer_nn : ilp_1pNw"
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --mem=200GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

echo "loading"
module load python/3.6.1
module load cudnn/v6
module load cuda/8.0.61
module load tensorflow/1.6.0-py36-gpu
echo "loaded"

if [ $1 = 2 ];
then
    python learn2.py --num $2
elif [ $1 = 4 ];
then
    python learn4.py --num $2
fi