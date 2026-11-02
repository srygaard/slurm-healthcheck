from dataclasses import asdict

from healthcheck.config import Gpu, GpuLimits, gpu_for, gpu_from_template, gpu_gres


def test_known_chips_fill_gpu_limits():
    assert gpu_for("h100:4") == Gpu(count=4, **asdict(gpu_gres["h100"]))
    assert gpu_for("b200:8") == Gpu(count=8, **asdict(gpu_gres["b200"]))
    assert gpu_for("l40s:4") == Gpu(count=4, **asdict(gpu_gres["l40s"]))
    assert gpu_for(None) == Gpu()
    assert gpu_for("4") == Gpu(count=4)


def test_sxm_link_and_ecc_limits():
    expected = GpuLimits(require_ecc=True, min_pcie_gen=5, min_pcie_width=16, min_nvlinks=18)
    assert gpu_gres["h100"] == expected
    assert gpu_gres["b200"] == expected
    assert expected.max_volatile_uncorrected_ecc == expected.max_remap_failures == expected.max_remap_pending == 0
    assert gpu_gres["l40s"] == GpuLimits(require_ecc=True, min_pcie_gen=4, min_pcie_width=16, min_nvlinks=0)


def test_gpu_from_template(monkeypatch, tmp_path):
    (tmp_path / "trillium.sh").write_text("#SBATCH --gpus-per-node=h100:4\n")
    (tmp_path / "trillium.b200.sh").write_text("#SBATCH --gpus-per-node=b200:8\n")
    (tmp_path / "rorqual.sh").write_text("#SBATCH --gpus-per-node=h100:4\n")
    monkeypatch.delenv("HC_TEMPLATE", raising=False)
    monkeypatch.setenv("CC_CLUSTER", "trillium")
    assert gpu_from_template(tmp_path) == gpu_for("h100:4")
    monkeypatch.setenv("HC_TEMPLATE", "trillium.b200")
    assert gpu_from_template(tmp_path) == gpu_for("b200:8")
    assert gpu_from_template(tmp_path, stem="trillium") == gpu_for("h100:4")
    monkeypatch.setenv("CC_CLUSTER", "rorqual")
    monkeypatch.delenv("HC_TEMPLATE")
    assert gpu_from_template(tmp_path) == gpu_for("h100:4")
    monkeypatch.setenv("CC_CLUSTER", "missing")
    assert gpu_from_template(tmp_path) == Gpu()
    monkeypatch.delenv("CC_CLUSTER")
    assert gpu_from_template(tmp_path) == Gpu()
