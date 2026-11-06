from dataclasses import replace

import pytest

from healthcheck.config import GpuLimits, gpu_gres, gpu_limits
from launcher.slurm import Gres


def test_sxm_link_and_ecc_limits():
    """The hardware fields. dcgm_timeout_s is tuned from measured suite times, so it is left out."""

    def part(name):
        return replace(gpu_gres[name], dcgm_timeout_s=0)

    sxm = GpuLimits(require_ecc=True, min_pcie_gen=5, min_pcie_width=16, min_nvlinks=18, dcgm_timeout_s=0)
    assert part("h100") == part("b200") == sxm
    assert sxm.max_volatile_uncorrected_ecc == sxm.max_remap_failures == sxm.max_remap_pending == 0
    pcie = GpuLimits(require_ecc=True, min_pcie_gen=4, min_pcie_width=16, min_nvlinks=0, dcgm_timeout_s=0)
    assert part("l40s") == pcie


def test_gpu_limits_rejects_a_missing_or_unknown_chip():
    gres, limits = gpu_limits(Gres("h100", 4))
    assert (gres, limits) == (Gres("h100", 4), gpu_gres["h100"])
    with pytest.raises(ValueError, match="unknown GPU type a100"):
        gpu_limits(Gres("a100", 8))
    with pytest.raises(ValueError, match="no named --gpus-per-node"):
        gpu_limits(None)
    with pytest.raises(ValueError, match="no named --gpus-per-node"):
        gpu_limits(Gres(None, 4))
