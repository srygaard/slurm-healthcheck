#!/bin/bash
# Lynx's RTX PRO 6000 nodes: run with CC_CLUSTER=lynx-rtx. 2x AMD EPYC 9755, one 200 Gb IB HCA, no NVLink.
#SBATCH --account=beta
#SBATCH --partition=gpubase_main_rtx,gpubase_interac_rtx
#SBATCH --exclusive
#SBATCH --gpus-per-node=rtx:8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=0
#SBATCH --time=00:35:00

source "$HC_CODE/slurm/scripts/.lynx-rtx.env"  # no Lmod: HC_MODULES is empty
