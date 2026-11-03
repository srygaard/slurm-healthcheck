import pytest

from healthcheck.config import GpuLimits, gpu_gres, gpu_limits
from launcher.slurm import Gres


def test_sxm_link_and_ecc_limits():
    expected = GpuLimits(require_ecc=True, min_pcie_gen=5, min_pcie_width=16, min_nvlinks=18)
    assert gpu_gres["h100"] == expected
    assert gpu_gres["b200"] == expected
    assert expected.max_volatile_uncorrected_ecc == expected.max_remap_failures == expected.max_remap_pending == 0
    assert gpu_gres["l40s"] == GpuLimits(require_ecc=True, min_pcie_gen=4, min_pcie_width=16, min_nvlinks=0)


def test_gpu_limits_rejects_a_missing_or_unknown_chip():
    gres, limits = gpu_limits(Gres("h100", 4))
    assert (gres, limits) == (Gres("h100", 4), gpu_gres["h100"])
    with pytest.raises(ValueError, match="unknown GPU type a100"):
        gpu_limits(Gres("a100", 8))
    with pytest.raises(ValueError, match="no named --gpus-per-node"):
        gpu_limits(None)
    with pytest.raises(ValueError, match="no named --gpus-per-node"):
        gpu_limits(Gres(None, 4))
