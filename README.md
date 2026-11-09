# slurm-healthcheck

A pre-flight health check for full-H100 nodes on Rorqual, run before a large multi-node training job.
Each job takes about four minutes, and the result is an `#SBATCH --exclude=` line for the nodes that fail
the checks (see [Configuration](#configuration)).

## Configuration

| file | sets | takes effect |
|------|------|--------------|
| [healthcheck/config.py](healthcheck/config.py) | thresholds, `homogeneity` keys, DCGM tests, NCCL timeouts, `/tmp` test size | next job; thresholds also on `report <run>` |
| [slurm/templates/*.sh](slurm/templates/) | GPU and count via `--gpus-per-node`: `h100:4` on Fir, Rorqual, Tamia and Trillium; `l40s:4` on Mila | next job; `report` grades that chip |
| [slurm/scripts/.drac.env](slurm/scripts/.drac.env), [.mila.env](slurm/scripts/.mila.env) | modules, nccl-tests version, CUDA arch | after `setup` |
| `healthcheck.py` options | which nodes, how they're grouped into jobs, `--exclude-on` | per command, see [Running](#running) |

Jobs import the code from this checkout, so an edit to `healthcheck/config.py` also reaches jobs that are still
queued. Run `setup` again after changing `uv.lock`, `.python-version` or `slurm/scripts/.drac.env`/`.mila.env`.

## Setup

Build on the login node, which can reach git and PyPI. Each healthcheck job skips DCGM when `dcgmi` is missing, or
when no hostengine answers and `nv-hostengine` is missing. It skips NCCL when `srun --mpi=list` offers neither
`pmix` nor `pmi2`.

```bash
uv run healthcheck.py setup               # nccl-tests, read-test files, venv
```
Anything already built is skipped.

## Running

`run` submits and returns. A job takes at least 9 minutes; run `report` when it finishes.

```bash
uv run healthcheck.py run                                      # one job across all eligible nodes
uv run healthcheck.py run --job-per switch                     # one job per leaf switch
uv run healthcheck.py run --job-per node --nodes rg31601       # one job per node, here a single node
uv run healthcheck.py run --switches ihpc316u25                # only nodes under these leaf switches
uv run healthcheck.py run --any 2                              # one job on any 2 free nodes picked by Slurm
uv run healthcheck.py run --dependency afterok:12345           # sbatch --dependency for every job of the run

uv run healthcheck.py report $SCRATCH/slurm-healthcheck/results/<job_id>                         # re-evaluate a finished run
uv run healthcheck.py report --exclude-on fail,warn $SCRATCH/slurm-healthcheck/results/<job_id>  # also exclude nodes that only warn
```

The job template's `#SBATCH` lines also decide the eligible nodes: exactly `gpu:<--gpus-per-node>` in
`sinfo %G` (so no MIG or partial-GPU nodes), all the `--constraint` features, in the `--partition` if set, and in
`USABLE_STATES` (launcher/selection.py). Trillium's fabric is non-blocking and Slurm has no switch topology there,
so `--job-per switch` fails.

`--job-per` also decides what the NCCL group and alltoall tests cover, since they run across all of a job's nodes
when it has more than two: every node with `cluster` (the default), one leaf switch with `switch`, and none with
`node`; with `--any N`, the N nodes Slurm picks, on whichever switches they are. Their time grows with the job, up
to `nccl.<kind>.timeout_s`.

## What a job checks
Each node runs three phases:

1. **Phase A**, all steps run in parallel; the job's time limit caps them:
   - **sysinfo**: GPU, NIC, NUMA, software, BIOS, and CPU power. The NUMA layout goes to `<run>/numa/<host>.json`.
   - **scratch**: `$SCRATCH` read; `scratch.enabled: false` skips it, and `setup` skips its files too.
   - **local_tmp**: `/tmp` free space and write/read, and SMART health (usually root-only).
   - **dcgm**: `dcgmi diag` with `dcgm.tests`, through the node's hostengine or one the job starts. A sh template
   drops tests with `HC_DCGM_SKIP_TESTS`.
   - **cpu_load**: busy loops on `cpu_load.busy_frac` of each CPU package's CPUs until dcgm returns. Per package:
     - **throttle**: warns on any thermal throttling episode during the full load.
     - **median**: warns if loaded CPU clock at full GPU power is under 90% of the run's median.
     - **sag**: warns if the median CPU clocks drop by 15% during full load. Fails when dropped by more than 40%.

     The clocks are graded only after 4 min at full GPU power ([Node Power Limit](#node-power-limit)).
2. **Phase B**, `all_reduce_perf` with 4 ranks per node:
   1. each node alone (NVLink);
   2. each node alone with P2P/SHM disabled (NIC loopback);
   3. ring pairs of nodes, so a bad NIC shows up in both of a node's pairs;
   4. all of the job's nodes together, when there are more than two: a leaf switch's with `--job-per switch`,
      every node with `--job-per cluster`.
3. **Phase C**: Xid events from a watcher that runs for the whole job, GPU error counter deltas, NCCL `WARN`
   lines, and kernel log matches when they are readable (not on Rorqual).

## Off by default
Turn these on in `healthcheck/config.py`.

- **project** (`project.enabled`): `$PROJECT` read, like scratch, with its own thresholds. `$PROJECT` is set in
  `slurm/scripts/.drac.env` and `.mila.env`. `setup` and `run` need `$PROJECT` set; while this is off, `setup`
  skips its files.
- **alltoall** (`nccl.alltoall.enabled`): `alltoall_perf` on the same nodes as the group test, every rank sending
  to every other. Its timeout is 600 s, so extend the template's `#SBATCH --time` first. Alone, with every other
  NCCL test disabled, it runs from two nodes.

The report fails a node when a check fails, when a step errors or times out, or when its result file is still
empty after the job. It also compares the `homogeneity` keys across nodes and flags those that differ from the
majority; with only two nodes, which one counts as the majority is arbitrary.

## Results
Every job writes to `$SCRATCH/slurm-healthcheck/results/<job_id>/`. A run with several jobs shares the directory
of its first job:
```
results/<job_id>/
  report.txt  report.json            outputs
  exclude.sbatch  include.sbatch     #SBATCH --exclude= failed nodes, --nodelist= passed nodes
  jobs.json                          submitted jobs, with states from sacct
  jobs/<job_id>-{stdout,stderr}.txt  job logs
  switches.json                      leaf switch -> its nodes, written at submit
  nodes/<host>.json                  per-node results
  groups/<job_id>.json               NCCL results; raw output in groups/<job_id>/
  logs/nccl/<test>-<host>-<pid>      NCCL debug logs (.log), kept only if a rank wrote one
  logs/dcgm/<host>/diag.txt          dcgmi diag stdout; diag.stderr.txt when the command wrote any
  numa/<host>.json                   per-node NUMA layout table
  xid/<host>.jsonl                   Xid events
```

## Finding the slow node of a pair
A bad node slows both of its ring pairs, so in a larger run it fails `nccl.pair` with `2/2 slow pairs` while its
neighbours only warn with `1/2`. Every NCCL kind fails below 0.9 of its median and warns below 0.95. On pairs the
fail line is what counts as slow, and a pair between the two lines warns both nodes. Two nodes form a single pair, so
a slow pair fails both with `1/1 slow pairs`. To
tell which one is slow:

1. Look at each node's own NCCL tests in the same report: a node that also fails `nccl.loopback` (its NIC alone) or
   `nccl.intra`, or has a `homogeneity` finding on `nic.ib[*].ports[*].rate` or a `nic.ports` finding, is the slow one. Their bandwidths are in
   `groups/<job_id>.json`.
2. If neither stands out, run the two again with a node that just passed (from a recent `include.sbatch`):
   ```bash
   uv run healthcheck.py run --job-per cluster --nodes rg21701,rg21702,rg31603
   ```
   The three form a ring of three pairs, each node in two. The slow node is the one both slow pairs share; read the
   pairs' `peak_busbw` in `groups/<job_id>.json` (~40 GB/s when healthy). The report may not flag it: two of the
   three pairs are slow, so the median is too, and only `nccl.pair.min_busbw_GBps` still applies.
3. NCCL debug logs, `logs/nccl/pair-<a>-<b>.<host>.<pid>.log`, are kept when a rank logged something, such as a
   `WARN` naming the failing connection.

## Node Power Limit
cpu_load compares the CPU clock only while the GPUs are at full power, which dcgm's `targeted_power` test sets. The
default gives ~280 s, just over `cpu_load.min_full_power_s` (240 s). Throttling events are counted over the whole burn.

| setting | default | longer |
|---|---|---|
| `targeted_power.test_duration` in `dcgm.parameters` | 260 | 600 |
| `#SBATCH --time` in `slurm/templates/<cluster>.sh` | 00:15:00 | 00:20:00 |
| job time, one node / a switch | ~7 / ~8 min | ~13 / ~15 min |

The CPU clocks may drop without a throttle event. The `cpu_load.sag` compares each package with its own clock before the stress test.

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
  load the compilers, CUDA, NCCL and MPI listed in `slurm/scripts/.drac.env`; see the
  [Rorqual docs](https://docs.alliancecan.ca/wiki/Rorqual/en).

## Tests
```bash
uv run pytest                            # unit tests and the doctests in launcher/ and healthcheck/
```
