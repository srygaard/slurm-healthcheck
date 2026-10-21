#!/bin/bash
# job_submit.lua appends _cpu/_gpu to the account
#SBATCH --account=rrg-bengioy-ad
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:10:00

source "$HC_CODE/slurm/scripts/.env"
module load $HC_MODULES $HC_BUILD_MODULES
