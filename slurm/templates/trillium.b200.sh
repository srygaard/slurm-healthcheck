#!/bin/bash
# Trillium B200. Select with HC_TEMPLATE=trillium.b200; trillium.sh stays the 4x H100 template.
# 15 min minimum walltime; multi-node jobs take a full 8-GPU node.
#SBATCH --account=rrg-bengioy-ad
#SBATCH --partition=compute_b200_full_node
#SBATCH --exclusive
#SBATCH --gpus-per-node=b200:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:30:00

source "$HC_CODE/slurm/scripts/.drac.env"
module load $HC_MODULES
