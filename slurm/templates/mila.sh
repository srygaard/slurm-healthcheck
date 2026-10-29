#!/bin/bash
# Mila: main's QOS caps a user at 8 CPUs and 2 GPUs and short-unkillable's at 4 GPUs, so only long takes whole nodes
#SBATCH --partition=long  # holds every A100L node; the 2 H100 nodes are only in staff-idt and short-unkillable
#SBATCH --exclusive
#SBATCH --gpus-per-node=a100l:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:15:00

source "$HC_CODE/slurm/scripts/.mila.env"
module load $HC_MODULES
