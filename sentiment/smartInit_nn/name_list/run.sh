#!/bin/bash
#SBATCH --get-user-env
#SBATCH --job-name="liu121"
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem=5GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --qos=express
echo "loading"
module load python/3.6.1
echo "loaded"

python nostop.py