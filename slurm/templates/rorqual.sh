#!/bin/bash
# job_submit.lua appends _cpu/_gpu to the account
#SBATCH --account=def-bengioy
#SBATCH --exclusive
#SBATCH --reservation=MILABENCH2026
#SBATCH --gpus-per-node=h100:4
#SBATCH --constraint=h100&nvlink
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=00:15:00

source "$HC_CODE/slurm/scripts/.drac.env"
module load $HC_MODULES
