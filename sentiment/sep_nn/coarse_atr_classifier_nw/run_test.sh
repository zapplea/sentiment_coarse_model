#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="liu121"
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --mem=5GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

echo "loading"
module load python/3.6.1
module load tensorflow/1.3.0-py35-gpu
echo "loaded"

echo run learn.py
python af_unittest.py
