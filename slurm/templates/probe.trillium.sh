#!/bin/bash
# Trillium: 15 min minimum walltime, GPU jobs only, multi-node jobs take full 4-GPU nodes
#SBATCH --account=rrg-bengioy-ad
#SBATCH --partition=compute_full_node  # trig0001 is only in debug_full_node
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:15:00

source "$HC_CODE/slurm/scripts/.env"
module load $HC_MODULES
