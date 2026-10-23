#!/bin/bash
# Tamia: GPU partitions are by-node only; the H200 nodes have gpu:h200:8, so gres alone selects the H100 nodes
#SBATCH --account=aip-bengioy
#SBATCH --partition=gpubase_bynode_b1  # 3 h max, holds every GPU node
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:10:00

source "$HC_CODE/slurm/scripts/.env"
module load $HC_MODULES
