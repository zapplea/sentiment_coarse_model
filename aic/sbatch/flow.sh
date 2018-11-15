#!/bin/bash

if test $1 = "cs";
then
    sbatch run_CoarseSentiNet.sh $2
fi