#!/bin/bash
# Trillium B200: the scinet reservation holds every node; b200-friendly-users overlaps it until 2026-12-03
#SBATCH --account=rrg-bengioy-ad
#SBATCH --partition=compute_b200_full_node
#SBATCH --reservation=b200-friendly-users
#SBATCH --exclusive
#SBATCH --gpus-per-node=b200:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=00:15:00

source "$HC_CODE/slurm/scripts/.drac.env"
module --version &>/dev/null || source /etc/profile  # Lmod; an exported module() may lack its helper
module load $HC_MODULES

