#!/bin/bash
# The cn-l nodes: each is 4x L40S, 48 CPUs, feature 48gb. main's QOS caps a user at 2 GPUs,
# so only long takes the whole node. Exclusive, so the node has to be free.
#SBATCH --partition=long
#SBATCH --exclusive
#SBATCH --gpus-per-node=l40s:4
#SBATCH --constraint=48gb
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=00:15:00

source "$HC_CODE/slurm/scripts/.mila.env"
module load $HC_MODULES
