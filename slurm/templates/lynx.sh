#!/bin/bash
#SBATCH --account=beta
#SBATCH --partition=gpubase_bynode_b0,gpubase_bynode_b2,gpubase_bynode_b3
#SBATCH --exclusive
#SBATCH --gpus-per-node=b200:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=112
#SBATCH --mem=0
#SBATCH --time=00:15:00

source "$HC_CODE/slurm/scripts/.lynx.env"  # no Lmod: HC_MODULES is empty
