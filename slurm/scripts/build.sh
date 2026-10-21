# Build job body, after the template loads the modules: nccl-tests, the shared venv and $SCRATCH read-test files.
# Runs on a compute node to keep compiles off the login node.
set -euo pipefail

echo "== nccl-tests $HC_NCCL_TESTS_TAG"
if [[ ! -x $HC_NCCL_TESTS/build/all_reduce_perf ]]; then
    rm -rf "$HC_NCCL_TESTS"
    mkdir -p "$(dirname "$HC_NCCL_TESTS")"
    git clone --depth 1 --branch "$HC_NCCL_TESTS_TAG" https://github.com/NVIDIA/nccl-tests.git "$HC_NCCL_TESTS"
    make -C "$HC_NCCL_TESTS" -j "${SLURM_CPUS_PER_TASK:-8}" MPI=1 MPI_HOME="$EBROOTOPENMPI" \
        CUDA_HOME="${CUDA_HOME:-$EBROOTCUDACORE}" NCCL_HOME="$EBROOTNCCL" NVCC_GENCODE="$HC_NVCC_GENCODE"
fi

echo "== venv $HC_VENV"
mkdir -p "$(dirname "$HC_VENV")"
export UV_PROJECT_ENVIRONMENT="$HC_VENV" UV_CACHE_DIR="$HC_BASE/.uv-cache" UV_PYTHON_DOWNLOADS=never
cd "$HC_CODE"
flock "$HC_VENV.lock" "$HC_UV" sync --frozen --no-dev --python "$(command -v python3)"

echo "== read-test files"
"$HC_VENV/bin/python" -m healthcheck.scratch
echo "== build done"
