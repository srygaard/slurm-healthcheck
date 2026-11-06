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
module load $HC_MODULES


# dcgm on 8x B200, each test alone (2026-11-05/06), with the GPUs' enforced power limit:
#   test             1000 W (trig0151)               200 W cap (trig0123, trig0152)
#   pcie             -                               247 s (job 1245724)
#   diagnostic       ~74 s                           74 s
#   targeted_power   ~260 s, pass (job 1248549)      skips at once (job 1248562)
#   memory           129 s, pass (job 1248636)       hangs to the dcgm timeout (jobs 1245047, 1248589)
#   targeted_stress  not timed                       33 s, fail: can't reach its stress level (job 1248635)
# pcie is skipped for time: NCCL intra and loopback cover GPU and NIC bandwidth; its PCIe replay check is lost. The
# rest take ~510 s on a healthy node, under the B200's dcgm_timeout_s of 650 s. Dcgm.skip_when_power_capped also
# drops memory on a capped node, where it hangs.
export HC_DCGM_SKIP_TESTS=pcie
