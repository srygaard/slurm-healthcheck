#!/bin/bash
# Fir: job_submit.lua appends _cpu/_gpu to the account and picks the partition; MIG nodes have h100mig gres types
#SBATCH --account=rrg-bengioy-ad
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus-per-node=h100:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:10:00

source "$HC_CODE/slurm/scripts/.env"
module load $HC_MODULES
