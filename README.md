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
uv run healthcheck.py run --mode cluster                       # one job across all eligible nodes
uv run healthcheck.py run --mode node --nodes rg31601          # one job per node, here a single node
uv run healthcheck.py run --switches ihpc316u25                # only nodes under these leaf switches
uv run healthcheck.py run --any 2                              # one job on any 2 free nodes picked by Slurm
uv run healthcheck.py run --no-wait                            # submit and exit; run `report` later
uv run healthcheck.py run --dependency afterok:12345           # sbatch --dependency for every job of the run

uv run healthcheck.py report $SCRATCH/slurm-healthcheck/results/<job_id>                         # re-evaluate a run
uv run healthcheck.py report --exclude-on fail,warn $SCRATCH/slurm-healthcheck/results/<job_id>  # also exclude nodes that only warn
```
The job template's `#SBATCH` lines also decide the eligible nodes: exactly `gpu:<--gpus-per-node>` in
`sinfo %G` (so no MIG or partial-GPU nodes), all the `--constraint` features, in the `--partition` if set, and in
`USABLE_STATES` (launcher/selection.py). Trillium's fabric is non-blocking and Slurm has no switch topology there,
so `--mode switch` runs as `--mode cluster`.
The cluster is detected from `$CC_CLUSTER` (or the `mila.quebec` domain), and each cluster needs its own job template,
`slurm/templates/<cluster>.sh`, which the probe and healthcheck jobs share. A template holds only the resources;
healthcheck.py passes the per-job name, logs, nodes, `--dependency` and the probe's `--nodes=1` as sbatch options.

The report is printed and written to the run directory. Paste one of its headers into the training job:
`exclude.sbatch` (`#SBATCH --exclude=rg[21703,31605]`) keeps the job off the failed nodes, and `include.sbatch`
(`#SBATCH --nodelist=rg[21701-21702,31601-31604]`) pins it to the nodes that passed. Both follow `--exclude-on`.
Slurm reports a healthcheck job as COMPLETED even when nodes fail; only the report says whether they passed.

## What a job checks
Each node runs three phases:

1. **Phase A**, four steps in parallel; the job's time limit caps them:
   - **sysinfo**: GPU driver, CUDA, ECC, remapped rows, PCIe, NVLink; IB firmware, port state and rate; MTU;
     NUMA nodes and each GPU's and HCA's PCI address, ids, NUMA node and PCIe root port, from which each node
     writes its NUMA layout table to `<run>/numa/<host>.json`;
     kernel, OS and OFED.
   - **scratch**: `$SCRATCH` read.
   - **local_tmp**: `/tmp` free space and write/read, and SMART health (usually root-only).
   - **dcgm**: `dcgmi diag` with the tests in `dcgm.tests` (software, memory, pcie, and the diagnostic,
     stress test shortened to 30 s, targeted_stress and targeted_power to 15 s), or a `dcgm.level`, through the node's
     hostengine, or one the job starts if the probe found none (`dcgm: host-self` in `env.json`).
2. **Phase B**, `all_reduce_perf` with 4 ranks per node:
   1. each node alone (NVLink);
   2. each node alone with P2P/SHM disabled (NIC loopback);
   3. ring pairs of nodes, so a bad NIC shows up in both of a node's pairs;
   4. the nodes under each leaf switch together, when there are more than two, all leaves at once.

   Each kind runs as one round of concurrent tests, so phase B takes about the same time at any node count. A test
   past its `nccl.timeout_s` is stopped and reported as `timeout`.
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
  switches.json                      node -> leaf switch, written at submit
  nodes/<host>.json                  per-node results, including the env.json they ran with
  groups/<job_id>.json               NCCL results; raw output in groups/<job_id>/
  logs/nccl/<test>-<host>-<pid>      NCCL debug logs (.log), kept only if a rank wrote one
  numa/<host>.json                   per-node NUMA layout table
  xid/<host>.jsonl                   Xid events
```
Setup jobs only have `jobs/`, plus the probe's DCGM and MPI test output.

## Calibrating thresholds
The medians at the top of `report.txt` are there to set the thresholds from. Typical values on Rorqual:

| median                 | measured | threshold                    |
|------------------------|----------|------------------------------|
| `nccl_intra_busbw`     | ~337     | `nccl.intra_min_busbw: 300`  |
| `nccl_loopback_busbw`  | ~6.4     | `nccl.loopback_min_busbw: 5` |
| `nccl_pair_busbw`      | ~40      | `nccl.pair_min_busbw: 20`    |
| `scratch_read_gbps`    | 0.05–0.3 | `scratch.read.min_gbps: 0.5`, warn only |

Bandwidths are GB/s. `$SCRATCH` read speed depends on the load on Lustre, which is why it only warns. Use
`gpu.min_nvlinks` and `nic.*` to pin the expected hardware. The thresholds are pydantic defaults in
`healthcheck/config.py`; after editing it, re-run `report <run>` instead of resubmitting.

## Tools Used

**NVIDIA**
- [DCGM diagnostics](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/dcgm-diagnostics.html):
  `dcgmi diag`, with `nv-hostengine` when the node doesn't run one.
- [nccl-tests](https://github.com/NVIDIA/nccl-tests) `all_reduce_perf`, configured with
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
