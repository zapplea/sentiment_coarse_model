#!/bin/bash
# --num $2 --dataset $3 --train_mod $4
sbatch run_learn.sh "learn4" 0 2 "v2"
sbatch run_learn.sh "learn4" 0 3 "v2"
sbatch run_learn.sh "learn4" 0 5 "v2"

sbatch run_learn.sh "learn4" 0 2 "v1"
sbatch run_learn.sh "learn4" 0 3 "v1"
sbatch run_learn.sh "learn4" 0 5 "v1"

sbatch run_learn.sh "learn2" 0 "v1"
sbatch run_learn.sh "learn2" 0 "v2"