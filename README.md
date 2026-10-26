# slurm-healthcheck

A pre-flight health check for full-H100 nodes on Rorqual, run before a large multi-node training job.
Each job takes about four minutes, and the result is an `#SBATCH --exclude=` line for the nodes that fail
the checks (see [Configuration](#configuration)).

## Configuration
| file | sets | takes effect |
|------|------|--------------|
| [healthcheck/config.py](healthcheck/config.py) | thresholds, `homogeneity` keys, DCGM tests, NCCL timeouts, `/tmp` test size | next job; thresholds also on `report <run>` |
| [slurm/scripts/.env](slurm/scripts/.env) | modules, nccl-tests version, CUDA arch | after `setup` |
| `healthcheck.py` options | which nodes, how they're grouped into jobs, `--exclude-on` | per command, see [Running](#running) |

Jobs import the code from this checkout, so an edit to `healthcheck/config.py` also reaches jobs that are still
queued. `$SCRATCH/slurm-healthcheck/env.json` is written by the probe; run `setup` again rather than editing it.

## Setup
```bash
uv run healthcheck.py setup               # build here (nccl-tests, read-test files, venv), then a probe job
cat $SCRATCH/slurm-healthcheck/env.json   # probe result: how to run DCGM, which srun --mpi plugin works
```
Run `setup` again after changing `uv.lock`, `.python-version` or `slurm/scripts/.env`; anything already
built is skipped.

## Running
```bash
uv run healthcheck.py run                                      # one job per leaf switch, wait, then report
uv run healthcheck.py run --job-per cluster                    # one job across all eligible nodes
uv run healthcheck.py run --job-per node --nodes rg31601       # one job per node, here a single node
uv run healthcheck.py run --switches ihpc316u25                # only nodes under these leaf switches
uv run healthcheck.py run --any 2                              # one job on any 2 free nodes picked by Slurm
uv run healthcheck.py run --no-wait                            # submit and exit; run `report` later
uv run healthcheck.py run --dependency afterok:12345           # sbatch --dependency for every job of the run

uv run healthcheck.py report $SCRATCH/slurm-healthcheck/results/<job_id>                         # re-evaluate a finished run
uv run healthcheck.py report --exclude-on fail,warn $SCRATCH/slurm-healthcheck/results/<job_id>  # also exclude nodes that only warn
```
The job template's `#SBATCH` lines also decide the eligible nodes: exactly `gpu:<--gpus-per-node>` in
`sinfo %G` (so no MIG or partial-GPU nodes), all the `--constraint` features, in the `--partition` if set, and in
`USABLE_STATES` (launcher/selection.py). Trillium's fabric is non-blocking and Slurm has no switch topology there,
so `--job-per switch` runs as `--job-per cluster`.

`--job-per` also decides what the NCCL group and alltoall tests cover, since they run across all of a job's nodes
when it has more than two: one leaf switch with `switch` (the default), every node with `cluster`, and none with
`node`; with `--any N`, the N nodes Slurm picks, on whichever switches they are. Their time grows with the job, up
to `nccl.timeout_s`.

The cluster is detected from `$CC_CLUSTER` (or the `mila.quebec` domain), and each cluster needs its own job template,
`slurm/templates/<cluster>.sh`, which the probe and healthcheck jobs share. A template holds only the resources;
healthcheck.py passes the per-job name, logs, nodes, `--dependency` and the probe's `--nodes=1` as sbatch options.

The report is printed and written to the run directory. Paste one of its headers into the training job:
`exclude.sbatch` (`#SBATCH --exclude=rg[21703,31605]`) keeps the job off the failed nodes, and `include.sbatch`
(`#SBATCH --nodelist=rg[21701-21702,31601-31604]`) pins it to the nodes that passed. Both follow `--exclude-on`.
Slurm reports a healthcheck job as COMPLETED even when nodes fail; only the report says whether they passed.

## What a job checks
Each node runs three phases:

1. **Phase A**, all steps run in parallel; the job's time limit caps them:
   - **sysinfo**: GPU driver, CUDA, ECC, remapped rows, PCIe, NVLink; IB firmware, port state and rate; MTU;
     NUMA nodes and each GPU's and HCA's PCI address, ids, NUMA node and PCIe root port, from which each node
     writes its NUMA layout table to `<run>/numa/<host>.json`;
     kernel, OS and OFED; system BIOS version (DMI), and the CPU clock and power settings a user can read
     (cpufreq governor and max clock, turbo, energy-performance preference, RAPL limits).
   - **scratch**: `$SCRATCH` read.
   - **local_tmp**: `/tmp` free space and write/read, and SMART health (usually root-only).
   - **dcgm**: `dcgmi diag` with the tests in `dcgm.tests` (software, memory, pcie, diagnostic, targeted_stress,
     targeted_power; stress tests cut to 15–30 s, diagnostic at `dcgm.diagnostic_precision`) or a `dcgm.level`,
     through the node's hostengine, or one the job starts if the probe found none (`dcgm: host-self` in
     `env.json`). A GPU fails below 90% of the run's median `diagnostic` TFLOPS and warns below 95%.
   - **cpu_load**: busy loops on `cpu_load.busy_frac` of the CPUs until dcgm returns (~2 min), without slowing the
     other steps. It flags a busy-core clock that drops more than 10% between the first and last 30 s or sits
     below 90% of the median node, and any CPU thermal throttling.
2. **Phase B**, `all_reduce_perf` with 4 ranks per node, then `alltoall_perf`:
   1. each node alone (NVLink);
   2. each node alone with P2P/SHM disabled (NIC loopback);
   3. ring pairs of nodes, so a bad NIC shows up in both of a node's pairs;
   4. all of the job's nodes together, when there are more than two: a leaf switch's with `--job-per switch`,
      every node with `--job-per cluster`;
   5. the same nodes with `alltoall_perf`, where every rank sends to every other.

   Each kind runs as one round of concurrent tests, so the first three take about the same time at any node count;
   the last two grow with the job. A test past its `nccl.timeout_s` is stopped and reported as `timeout`.
3. **Phase C**: Xid events from a watcher that runs for the whole job, GPU error counter deltas, NCCL `WARN`
   lines, and kernel log matches when they are readable (not on Rorqual).

The report fails a node when a check fails, when a step errors or times out, or when its result file is still
empty after the job. It also compares the `homogeneity` keys across nodes and flags those that differ from the
majority; with only two nodes, which one counts as the majority is arbitrary.

## Results
Every job, setup jobs included, writes to `$SCRATCH/slurm-healthcheck/results/<job_id>/`. A run with several
jobs shares the directory of its first job:
```
results/<job_id>/
  report.txt  report.json            outputs
  exclude.sbatch  include.sbatch     #SBATCH --exclude= failed nodes, --nodelist= passed nodes
  jobs.json                          submitted jobs, with states from sacct
  jobs/<job_id>-{stdout,stderr}.txt  job logs
  switches.json                      leaf switch -> its nodes, written at submit
  nodes/<host>.json                  per-node results, including the env.json they ran with
  groups/<job_id>.json               NCCL results; raw output in groups/<job_id>/
  logs/nccl/<test>-<host>-<pid>      NCCL debug logs (.log), kept only if a rank wrote one
  numa/<host>.json                   per-node NUMA layout table
  xid/<host>.jsonl                   Xid events
```
Setup jobs only have `jobs/`, plus the probe's DCGM and MPI test output.

## Calibrating thresholds
The medians at the top of `report.txt` are there to set the thresholds from. Typical values on Rorqual:

| median                     | measured | threshold                         |
|----------------------------|----------|-----------------------------------|
| `dcgm_diagnostic_tflops`   | ~650     | per GPU, `dcgm.diagnostic_min_frac_of_median: 0.90` of it, warn at 0.95 |
| `nccl_intra_busbw_GBps`    | ~337     | `nccl.intra_min_busbw_GBps: 300`  |
| `nccl_loopback_busbw_GBps` | ~6.4     | `nccl.loopback_min_busbw_GBps: 5` |
| `nccl_pair_busbw_GBps`     | ~40      | `nccl.pair_min_busbw_GBps: 20`    |
| `nccl_group_busbw_GBps`    | ~24.6    | `nccl.group_min_busbw_GBps: 20`   |
| `nccl_alltoall_busbw_GBps` | ~6.8     | `nccl.alltoall_min_busbw_GBps: 5` |
| `scratch_read_GBps`        | 0.05–0.3 | `scratch.read.min_GBps: 0.2`, warn only |

The group and alltoall values are for one leaf switch (7 nodes, `--job-per switch`). Across switches the traffic
goes through the spine: 21 nodes on 3 switches with `--job-per cluster` measured ~19.2 and ~3.0, below these
thresholds, so such a run warns on every node.

Bandwidths are GB/s. TFLOPS are DCGM's coarse estimate at `dcgm.diagnostic_precision: fp16`: each H100 reads
640.9 or 659.8 (runs before fp16 used DCGM's default of all three precisions and read 160–165). `$SCRATCH` read
speed depends on the load on Lustre, which is why it only warns. Use `gpu.min_nvlinks` and `nic.*` to pin the
expected hardware. The thresholds are pydantic defaults in
`healthcheck/config.py`; after editing it, re-run `report <run>` instead of resubmitting.

## Finding the slow node of a pair
A bad node slows both of its ring pairs, so in a larger run it fails `nccl.pair` with `2/2 slow pairs` while its
neighbours only warn with `1/2`. Two nodes form a single pair, so a slow pair fails both with `1/1 slow pairs`. To
tell which one is slow:

1. Look at each node's own NCCL tests in the same report: a node that also fails `nccl.loopback` (its NIC alone) or
   `nccl.intra`, or has `nic.rate` or `nic.ports` findings, is the slow one. Their bandwidths are in
   `groups/<job_id>.json`.
2. If neither stands out, run the two again with a node that just passed (from a recent `include.sbatch`):
   ```bash
   uv run healthcheck.py run --job-per cluster --nodes rg21701,rg21702,rg31603
   ```
   The three form a ring of three pairs, each node in two. The slow node is the one both slow pairs share; read the
   pairs' `peak_busbw` in `groups/<job_id>.json` (~40 GB/s when healthy). The report may not flag it: two of the
   three pairs are slow, so the median is too, and only `nccl.pair_min_busbw_GBps` still applies.
3. NCCL debug logs, `logs/nccl/pair-<a>-<b>.<host>.<pid>.log`, are kept when a rank logged something, such as a
   `WARN` naming the failing connection.

## Heat Soak
A default job loads CPUs and GPUs together for only ~2 minutes. cpu_load runs until dcgm returns, so lengthening dcgm's last test, `targeted_power` (GPUs at
their power cap), lengthens the combined load:

| setting | default | soak |
|---|---|---|
| `targeted_power.test_duration` in `dcgm.parameters` | 15 | 600 |
| `#SBATCH --time` in `slurm/templates/<cluster>.sh` | 00:10:00 | 00:20:00 |
| `cpu_load.window_s`, optional, for steadier windows | 30 | 60 |

A one-node soak takes ~13 minutes, plus ~2 for phase B across nodes.

## Tools Used

**NVIDIA**
- [DCGM diagnostics](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html):
  `dcgmi diag`, with `nv-hostengine` when the node doesn't run one.
- [nccl-tests](https://github.com/NVIDIA/nccl-tests) `all_reduce_perf` and `alltoall_perf`, configured with
  [NCCL environment variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html).
- [NVML](https://docs.nvidia.com/deploy/nvml-api/index.html) through
  [nvidia-ml-py](https://pypi.org/project/nvidia-ml-py/): GPU configuration, error counters and Xid events
  ([Xid reference](https://docs.nvidia.com/deploy/xid-errors/index.html)).
- [nvidia-smi](https://docs.nvidia.com/deploy/nvidia-smi/index.html) `topo -m` for the GPU/NIC topology.
- [MLNX_OFED](https://docs.nvidia.com/networking/display/mlnxofedv24100700) `ofed_info` for the InfiniBand
  driver version.

**System**
- [smartctl](https://www.smartmontools.org/) and [lsblk](https://man7.org/linux/man-pages/man8/lsblk.8.html)
  for local disk health.
- [dmesg](https://man7.org/linux/man-pages/man1/dmesg.1.html) and
  [journalctl](https://www.freedesktop.org/software/systemd/man/latest/journalctl.html) for kernel log matches.
- [Environment modules](https://docs.alliancecan.ca/wiki/Utiliser_des_modules/en) ([Lmod](https://lmod.readthedocs.io/))
  load the compilers, CUDA, NCCL and MPI listed in `slurm/scripts/.env`; see the
  [Rorqual docs](https://docs.alliancecan.ca/wiki/Rorqual/en).

## Tests
```bash
uv run pytest                            # unit tests and the doctests in launcher/ and healthcheck/
```
