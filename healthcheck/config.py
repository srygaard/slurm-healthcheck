"""The checks' settings and failure thresholds, and where a run's files live: results/<run id>/.

Bandwidths are GB/s, sizes GiB, durations seconds. Starting values are placeholders:
calibrate them from the medians in report.txt after a first `--job-per switch` run.
"""

import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from launcher.slurm import Gres


class Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


@dataclass(frozen=True, kw_only=True)
class GpuLimits:
    """The GPU fields that describe the part rather than a measured floor. The count comes from JobTemplate."""

    require_ecc: bool
    min_pcie_gen: int  # max link gen (current gen drops at idle)
    min_pcie_width: int
    min_nvlinks: int
    # How long dcgmi diag may run before it counts as hung: the full dcgm.tests suite on a healthy node plus ~20-40%.
    # The job template's --time must also fit NCCL after it.
    dcgm_timeout_s: float
    max_volatile_uncorrected_ecc: int = 0
    max_remap_failures: int = 0
    max_remap_pending: int = 0
    min_power_limit_frac: float = 0.95


# Keyed by the Slurm GRES type in --gpus-per-node. H100 and B200 are PCIe Gen5 x16 with 18 NVLinks.
# L40S is a PCIe card: Gen4 x16, no NVLink. ECC is on; row remap is an HBM counter and stays at zero.
gpu_gres: dict[str, GpuLimits] = {
    # 4x H100: 367 s on Trillium, ~7 min on Rorqual
    "h100": GpuLimits(require_ecc=True, min_pcie_gen=5, min_pcie_width=16, min_nvlinks=18, dcgm_timeout_s=540),
    # 8x B200: 516 s on Trillium, ~35 s of it pcie
    "b200": GpuLimits(require_ecc=True, min_pcie_gen=5, min_pcie_width=16, min_nvlinks=18, dcgm_timeout_s=650),
    "l40s": GpuLimits(require_ecc=True, min_pcie_gen=4, min_pcie_width=16, min_nvlinks=0, dcgm_timeout_s=600),
}


def gpu_limits(gres: Gres | None) -> tuple[Gres, GpuLimits]:
    """The named `--gpus-per-node` and its limits. ValueError when it is missing or not a known chip."""
    if gres is None or gres.name is None:
        raise ValueError("job template has no named --gpus-per-node")
    try:
        return gres, gpu_gres[gres.name]
    except KeyError:
        known = ", ".join(sorted(gpu_gres))
        raise ValueError(f"unknown GPU type {gres.name}; expected one of {known}") from None


class Nic(Section):
    """Port rates are compared to the other nodes and may fail on homogeneity."""

    min_active_ib_ports: int = 1
    link_layer: str = "InfiniBand"


class Scratch(Section):
    """The shared filesystem: load isn't a node fault, so this only warns by default."""

    enabled: bool = True  # off: setup skips its read-test files too
    files: int = 8  # read-test files written by setup; each node reads f<crc32(host) % n>
    file_gib: float = 1  # small: a busy OST writes at only a few MB/s
    write_block_mib: int = 16  # Lustre is slow with small writes
    read_max_s: float = 5  # loops over the file for read_max_s
    read_block_mib: int = 16
    min_GBps: float = 0.2
    min_frac_of_median: float = 0.5
    severity: Literal["fail", "warn"] = "warn"


class Project(Scratch):
    """$PROJECT, tested like $SCRATCH: load isn't a node fault, so this only warns by default."""

    enabled: bool = False  # off: no $PROJECT needed, and setup skips its files


class LocalTmpRw(Section):
    gib: float = 2
    max_s: float = 5  # each of write and read
    block_mib: int = 16
    min_write_GBps: float = 0.5
    min_read_GBps: float = 1.0


class LocalTmp(Section):
    """The node's own disk."""

    paths: list[str] = ["/tmp"]  # benchmarked once per filesystem
    rw: LocalTmpRw = LocalTmpRw()
    min_free_gib: float = 100
    fail_on_smart: bool = True


class Dcgm(Section):
    enabled: bool = True
    # DCGM test duration in seconds (test_duration parameter in parentheses)
    #                    4x H100     8x B200
    #   memory           27          129
    #   pcie (P2P=on)    21          226
    #   pcie (P2P=off)   ?           49
    #   diagnostic       53 (30)     76 (30)
    #   targeted_stress  38 (30)     31 (15)
    #   targeted_power   39 (30)     276 (260)
    #   nvbandwidth      227         ?
    # software (~1 s) always runs first and can't be named here. Without targeted_power, dcgm returns before
    # cpu_load.min_full_power_s and cpu_load grades no clocks.
    tests: list[str] = [
        "memory",  # allocates ~98% of each GPU's memory and checks it
        "pcie",  # host<->GPU bandwidth and PCIe replay errors
        # GEMM stress at diagnostic_precision, checks results and throttling. fp16 alone takes as long as DCGM's
        # default of all three precisions: this dcgm step ran 127 s either way on rg31603.
        "diagnostic",
        "targeted_stress",  # holds a target SM load
        "targeted_power",  # holds a target power draw
        # "nvbandwidth",  # GPU copy bandwidth; pcie and the NCCL intra test cover it
        # "nccl_tests",  # not timed; the NCCL step covers collectives
    ]
    port: int = 5556  # for a hostengine we start ourselves
    discovery_timeout_s: float = 15  # a missing hostengine errors at once; a hung discovery must not stall phase A
    # Tests dropped when a GPU's enforced power limit is under GpuLimits.min_power_limit_frac of its default: memory
    # hangs to the dcgm timeout on a B200 capped at 200 W.
    skip_when_power_capped: list[str] = ["memory"]
    parameters: list[str] = [  # dcgmi diag -p, joined with ";"
        "software.require_persistence_mode=false",  # Rorqual leaves it off and only root can enable it: not a node fault
        "diagnostic.test_duration=30",  # the stress tests default to minutes each
        "targeted_stress.test_duration=15",
        "targeted_power.test_duration=260",  # ~280 s at full GPU power with diagnostic: over cpu_load.min_full_power_s
        # pcie's GPU-to-GPU copies, with P2P on and off for every pair: 177 of its 226 s on 8x B200. NCCL intra covers
        # the GPU-to-GPU bandwidth.
        "pcie.test_p2p_on=false",
        "pcie.test_p2p_off=false",
    ]
    # diagnostic's GEMM precision (DCGM 4.7 has no BF16 or TF32). On rg31603's H100s: fp16 641-660 TFLOPS, fp32 170,
    # fp64 52; DCGM's default runs all three and reports a blend (160-165). fp16 is closest to BF16 training: H100 runs
    # FP16 and BF16 on the same tensor cores at the same peak.
    diagnostic_precision: Literal["fp64", "fp32", "fp16"] = "fp16"
    # diagnostic's per-GPU TFLOPS against the run's median GPU. DCGM's estimate is coarse: at fp16, 640.9 or 659.8.
    diagnostic_min_frac_of_median: float = 0.90  # fail below
    diagnostic_warn_frac_of_median: float = 0.95  # warn below

    @property
    def diagnostic_precision_letter(self) -> str:
        """diagnostic.precision takes one letter per GEMM; it logs anything else as invalid and runs its defaults."""
        return {"fp64": "d", "fp32": "s", "fp16": "h"}[self.diagnostic_precision]


class CpuLoad(Section):
    """Busy loops on a share of the CPUs while dcgm loads the GPUs; see healthcheck/cpu_load.py.

    It runs alongside every phase A step, holding until dcgm, the longest, returns (~6 min), so it adds no time.
    """

    enabled: bool = True
    busy_frac: float = 0.75  # of the job's CPUs, at SCHED_IDLE so the other steps still get CPU time when they need it
    interval_s: float = 1  # at 5 s the median of a ~4 min full-power window shifted with the sampling phase
    full_power_frac: float = 0.9  # samples at this share of the run's peak GPU power count as full power
    min_full_power_s: float = 240  # time at full GPU power needed before the clocks are graded
    min_frac_of_median: float = 0.9  # a package's busy-core clock at full GPU power, against the run's packages' median
    # A package's busy-core clock at full GPU power, against its own clock while the GPUs draw under
    # baseline_power_frac of their peak. trig0151 (8x B200) went from 3200 to ~2350 MHz at 7.97 kW with
    # no throttle event; the run's median can't see that on one node, or when every node sags alike.
    baseline_power_frac: float = 0.5
    warn_freq_frac_of_baseline: float = 0.85
    fail_freq_frac_of_baseline: float = 0.6


class NcclTest(Section):
    """Settings shared by every NCCL test kind: intra, loopback, pair, group, alltoall."""

    name: str = ""  # the kind recorded in results: intra, loopback, pair, group, alltoall
    enabled: bool = True
    binary: str = "all_reduce_perf"  # file in $HC_NCCL_TESTS/build
    args: list[str] = ["--minbytes=128M", "--maxbytes=1G", "--stepfactor=2", "--warmup_iters=5", "--iters=20"]
    env: dict[str, str] = Field(default_factory=dict)
    # A round's stragglers are terminated and reported as timeout, so a hung test can't use up the job and skip phase C.
    timeout_s: float = 120
    min_busbw_GBps: float = 0
    min_frac_of_median: float | None = 0.9
    warn_frac_of_median: float | None = 0.95
    # Severity of an unblamed timeout of a whole-job test. A slow result still warns.
    timeout_severity: Literal["fail", "warn"] = "fail"


class Nccl(Section):
    """One NcclTest per kind. Turn a kind off with enabled=false.

    Timeouts are about 3x the healthy times on Rorqual (intra 11 s, loopback 35 s, pair 15 s, a 21-node group 24 s,
    a leaf alltoall 58 s). A node failing its socket connections took ~140 s per pair. alltoall_perf's busbw is
    algbw * (n-1)/n and all_reduce_perf's is 2 * algbw * (n-1)/n, so the two don't compare.
    """

    debug: str = "WARN"
    intra: NcclTest = NcclTest(
        name="intra",
        timeout_s=45,
        min_busbw_GBps=120,  # single-node ~337; a 65-node run sat near 149, with two nodes near 20
    )
    loopback: NcclTest = NcclTest(
        name="loopback",
        env={
            "NCCL_P2P_DISABLE": "1",  # GPUs would otherwise talk over NVLink
            "NCCL_SHM_DISABLE": "1",  # the same-node fallback is host shared memory, still off the NIC
            "NCCL_NVLS_ENABLE": "0",  # NVLink SHARP would still carry the reduction
        },
        timeout_s=90,
        min_busbw_GBps=3,  # single-node ~6.4; the same 65-node run sat near 3.7, lowest 3.22
    )
    pair: NcclTest = NcclTest(name="pair", timeout_s=45, min_busbw_GBps=20)
    group: NcclTest = NcclTest(name="group", timeout_s=120, min_busbw_GBps=20)
    alltoall: NcclTest = NcclTest(
        name="alltoall",
        # Off: a 65-node alltoall was still starting at 180 s, and this 600 s timeout does not fit in the template's
        # --time once phase A has run. Extend #SBATCH --time before setting enabled=true.
        enabled=False,
        binary="alltoall_perf",
        # all_reduce walks 128 MB to 1 GB. alltoall replaces args: one 1 GB size, because a 65-node sweep spent its
        # timeout in communicator init before any size row.
        args=["--minbytes=1G", "--maxbytes=1G", "--warmup_iters=2", "--iters=5"],
        timeout_s=600,
        # A 64-node run, one 1 GB size: 1.97 GB/s. The floor is about half of that. A leaf was ~6.8; a leaf slow
        # against the other jobs in the run is caught by min_frac_of_median. Such a timeout names no node, so it warns.
        min_busbw_GBps=1.0,
        timeout_severity="warn",
    )


class Logs(Section):
    xid_watch: bool = True
    xid_ignore: set[int] = {  # application-induced Xids
        13,  # graphics engine exception: a kernel's illegal instruction or out-of-range access
        31,  # GPU memory page fault: an application's illegal memory access
        43,  # GPU stopped processing: the channel hit a software fault and was reset
        45,  # preemptive cleanup: channel torn down after a killed process or an earlier error
    }
    nccl_fail_patterns: list[str] = ["NCCL WARN", "completion with error", "Timeout", "remote process exited"]
    kernel_patterns: list[str] = ["NVRM", "Xid", "mlx5", "nvidia-peermem", "EDAC", "I/O error", "nvme"]
    # Matched kernel lines that warn instead of failing the node. Other kernel_patterns matches still fail.
    kernel_warn: list[str] = [
        "Failed to send inband data: 0",  # NVRM knvlink teardown at the end of each NCCL round; status 0
        "query_mcia_reg failed",  # mlx5 module query; on these nodes it was the disabled port 0000:d8:00.1
    ]
    max_nvlink_error_delta: int = 0


class Homogeneity(Section):
    """Key paths into sysinfo that should match on every node; [*] spans lists."""

    fail: list[str] = [
        "gpu.driver_version",
        "gpu.cuda_version",
        "gpu.devices[*].name",
        "nic.ib[*].fw_ver",
        "nic.ib[*].ports[*].rate",
    ]
    warn: list[str] = [
        # "gpu.devices[*].vbios",
        # "kernel",
        "os_release",
        "gpu.topo_hash",
        "nic.nvidia_peermem",
        "nic.net.ib0.mtu",
        "software.ofed",
        # rg21701 lost ~30% of its CPU clock under training load yet passed every check; only its BIOS differed
        "bios.bios_version",
        "cpu_power.governor",
        "cpu_power.max_mhz",
        "cpu_power.scaling_max_mhz",
        "cpu_power.epp",
        "cpu_power.no_turbo",
        "cpu_power.boost",
        "cpu_power.rapl",
    ]


class Host(Section):
    name: str = Field(default_factory=lambda: socket.gethostname().split(".")[0])


def _run_root() -> Path:
    """results/<job id of the run's first job>; healthcheck.py exports HC_RUN_ID to the run's later jobs. "." outside a job."""
    run_id = os.environ.get("HC_RUN_ID") or os.environ.get("SLURM_JOB_ID")
    return Path(os.environ.get("HC_BASE", ".")) / "results" / run_id if run_id else Path(".")


class Paths(Section):
    """The layout of a run's results directory, shared by the job steps and the report."""

    base: Path = Path(
        os.environ.get("HC_BASE", ".")
    )  # shared across runs: .opt/, diskprobe/, results/; healthcheck.py exports it
    root: Path = Field(default_factory=_run_root)
    project: Path = Path(os.environ.get("HC_PROJECT", "."))  # like base, on $PROJECT; slurm/scripts/.*.env export it

    @property
    def switches(self) -> Path:
        """Each leaf switch, mapped to the nodes the run targets under it."""
        return self.root / "switches.json"

    @property
    def jobs(self) -> Path:
        return self.root / "jobs.json"

    @property
    def job_scripts(self) -> Path:
        return self.root / "jobs"

    @property
    def nodes(self) -> Path:
        return self.root / "nodes"

    @property
    def groups(self) -> Path:
        return self.root / "groups"

    @property
    def nccl_logs(self) -> Path:
        return self.root / "logs" / "nccl"

    def node(self, host: str) -> Path:
        return self.nodes / f"{host}.json"

    def group(self, job_id: str) -> Path:
        return self.groups / f"{job_id}.json"

    def group_logs(self, job_id: str) -> Path:
        return self.groups / job_id

    def numa(self, host: str) -> Path:
        """The node's NUMA layout table, from its sysinfo."""
        return self.root / "numa" / f"{host}.json"

    def xid(self, host: str) -> Path:
        return self.root / "xid" / f"{host}.jsonl"

    def dcgm_logs(self, host: str) -> Path:
        return self.root / "logs" / "dcgm" / host


class Config(Section):
    nic: Nic = Nic()
    scratch: Scratch = Scratch()
    project: Project = Project()
    local_tmp: LocalTmp = LocalTmp()
    dcgm: Dcgm = Dcgm()
    cpu_load: CpuLoad = CpuLoad()
    nccl: Nccl = Nccl()
    logs: Logs = Logs()
    homogeneity: Homogeneity = Homogeneity()
    host: Host = Host()
    paths: Paths = Paths()


cfg = Config()
