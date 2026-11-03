from healthcheck.config import GpuLimits, gpu_gres


def test_sxm_link_and_ecc_limits():
    expected = GpuLimits(require_ecc=True, min_pcie_gen=5, min_pcie_width=16, min_nvlinks=18)
    assert gpu_gres["h100"] == expected
    assert gpu_gres["b200"] == expected
    assert expected.max_volatile_uncorrected_ecc == expected.max_remap_failures == expected.max_remap_pending == 0
    assert gpu_gres["l40s"] == GpuLimits(require_ecc=True, min_pcie_gen=4, min_pcie_width=16, min_nvlinks=0)
