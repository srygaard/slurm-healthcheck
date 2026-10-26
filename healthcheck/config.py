"""The checks' settings and failure thresholds, and where a run's files live: results/<run id>/.

Bandwidths are GB/s, sizes GiB, durations seconds. Starting values are placeholders:
calibrate them from the medians in report.txt after a first `--job-per switch` run.
"""

import os
import socket
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Gpu(Section):
    count: int = 4  # also the nccl-tests ranks per node
    require_ecc: bool = True
    max_volatile_uncorrected_ecc: int = 0
    max_remap_failures: int = 0
    max_remap_pending: int = 0
    min_pcie_gen: int = 5  # max link gen (current gen drops at idle)
    min_pcie_width: int = 16
    min_nvlinks: int = 18


class Nic(Section):
    min_active_ib_ports: int = 1
    min_rate_Gbps: float = 200
    link_layer: str = "InfiniBand"


class ScratchRead(Section):
    max_s: float = 5  # loops over the file for max_s
    block_mib: int = 16
    min_GBps: float = 0.2
    min_frac_of_median: float = 0.5
    severity: Literal["fail", "warn"] = "warn"


class Scratch(Section):
    """The shared filesystem: load isn't a node fault, so this only warns by default."""

    files: int = 8  # read-test files written by setup; each node reads f<crc32(host) % n>
    file_gib: float = 1  # small: a busy OST writes at only a few MB/s
    write_block_mib: int = 16  # Lustre is slow with small writes
    read: ScratchRead = ScratchRead()


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
    level: int = 2  # dcgmi diag -r: 1 software; 2 adds memory, pcie; 3 and 4 add long stress tests
    # Test names to run instead of level (DCGM 4.7); software (~1 s) always runs first and can't be named.
    # Times measured alone on rg32401, 4x H100, with the test_duration parameters below.
    tests: list[str] = [
        "memory",  # 27 s: allocates ~98% of each GPU's memory and checks it
        "pcie",  # 21 s: host<->GPU bandwidth and PCIe replay errors
        # 53 s: GEMM stress at diagnostic_precision, checks results and throttling. fp16 alone takes as long as DCGM's
        # default of all three precisions: this dcgm step ran 127 s either way on rg31603.
        "diagnostic",
        "targeted_stress",  # 38 s at test_duration=30: holds a target SM load
        "targeted_power",  # 39 s at test_duration=30: holds a target power draw (~610 W average, 700 W peak)
        # "nvbandwidth",  # 227 s: GPU copy bandwidth; pcie and the NCCL intra test cover it
        # "nccl_tests",  # not timed; the NCCL step covers collectives
    ]
    port: int = 5556  # for a hostengine we start ourselves
    parameters: list[str] = [  # dcgmi diag -p, joined with ";"
        "software.require_persistence_mode=false",  # Rorqual leaves it off and only root can enable it: not a node fault
        "diagnostic.test_duration=30",  # the stress tests default to minutes each
        "targeted_stress.test_duration=15",
        "targeted_power.test_duration=15",
    ]
    fail_on: list[str] = ["Fail"]
    warn_on: list[str] = ["Warn"]
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

    It runs alongside every phase A step, holding until dcgm, the longest, returns (~2 min), so it adds no time.
    """

    enabled: bool = True
    busy_frac: float = 0.75  # of the job's CPUs, at SCHED_IDLE so the other steps still get CPU time when they need it
    interval_s: float = 5
    full_power_frac: float = 0.9  # samples at this share of the run's peak GPU power count as full power
    min_full_power_s: float = 60  # ignores failures on (noisy) short 60s tests
    low_frac: float = 0.10  # a busy-core reading at full GPU power this far below the run's highest counts as low
    max_low_share: float = 0.2  # of those readings; soaks: rg21701 31-48% low, healthy nodes 0-8%
    min_frac_of_median: float = 0.9  # busy-core clock at full GPU power, against the median of the run's nodes
    severity: Literal["fail", "warn"] = "warn"


class Nccl(Section):
    binary: str = "all_reduce_perf"
    alltoall_binary: str = "alltoall_perf"  # the alltoall tests', built alongside binary
    args: list[str] = ["--minbytes=128M", "--maxbytes=1G", "--stepfactor=2", "--warmup_iters=5", "--iters=20"]
    debug: str = "WARN"
    # A round's stragglers are terminated and reported as timeout, so a hung test can't use up the job and skip
    # phase C. About 3x the healthy times on Rorqual (intra 11 s, loopback 35 s, pair 15 s, 21-node group 24 s and
    # alltoall 58 s; the last two grow with the job); a node failing its socket connections took ~140 s per pair.
    timeout_s: dict[str, float] = {"intra": 45, "loopback": 90, "pair": 45, "group": 120, "alltoall": 180}
    # The loopback round: no NVLink or shared memory, so traffic goes through the NICs
    loopback_env: dict[str, str] = {"NCCL_P2P_DISABLE": "1", "NCCL_SHM_DISABLE": "1", "NCCL_NVLS_ENABLE": "0"}
    intra_min_busbw_GBps: float = 300
    loopback_min_busbw_GBps: float = 5  # Rorqual: one 200 Gb/s NIC per node shared by 4 GPUs, ~6.3 at line rate
    pair_min_busbw_GBps: float = 20
    pair_min_frac_of_median: float = 0.7
    group_min_busbw_GBps: float = 20
    group_min_frac_of_median: float = 0.8
    # alltoall_perf's busbw is algbw * (n-1)/n, all_reduce_perf's 2 * algbw * (n-1)/n, so the two don't compare
    alltoall_min_busbw_GBps: float = 5  # Rorqual: ~6.8 within a leaf switch, ~3.0 across three
    alltoall_min_frac_of_median: float = 0.8


class Logs(Section):
    xid_watch: bool = True
    xid_ignore: list[int] = [  # application-induced Xids
        13,  # graphics engine exception: a kernel's illegal instruction or out-of-range access
        31,  # GPU memory page fault: an application's illegal memory access
        43,  # GPU stopped processing: the channel hit a software fault and was reset
        45,  # preemptive cleanup: channel torn down after a killed process or an earlier error
    ]
    nccl_fail_patterns: list[str] = ["NCCL WARN", "completion with error", "Timeout", "remote process exited"]
    kernel_patterns: list[str] = ["NVRM", "Xid", "mlx5", "nvidia-peermem", "EDAC", "I/O error", "nvme"]
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
    )  # shared across runs: env.json, .opt/, diskprobe/, results/; healthcheck.py exports it
    root: Path = Field(default_factory=_run_root)

    @property
    def env(self) -> Path:
        """Written by the probe job; each node's result copies what it read."""
        return self.base / "env.json"

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

    def xid_stop(self, job_id: str) -> Path:
        return self.root / "xid" / f".stop-{job_id}"

    def dcgm_logs(self, host: str) -> Path:
        return self.root / "logs" / "dcgm" / host


class Config(Section):
    gpu: Gpu = Gpu()
    nic: Nic = Nic()
    scratch: Scratch = Scratch()
    local_tmp: LocalTmp = LocalTmp()
    dcgm: Dcgm = Dcgm()
    cpu_load: CpuLoad = CpuLoad()
    nccl: Nccl = Nccl()
    logs: Logs = Logs()
    homogeneity: Homogeneity = Homogeneity()
    host: Host = Host()
    paths: Paths = Paths()


cfg = Config()
