# Run by `healthcheck.py setup` on the login node, since compute nodes may have no internet:
# nccl-tests, the shared venv and $SCRATCH and $PROJECT read-test files.
set -euo pipefail

type module &>/dev/null || source /etc/profile  # Lmod
source "$HC_CODE/slurm/scripts/.env"
module load $HC_MODULES

echo "== nccl-tests $HC_NCCL_TESTS_TAG"
if [[ ! -x $HC_NCCL_TESTS/build/all_reduce_perf ]]; then
    rm -rf "$HC_NCCL_TESTS"
    mkdir -p "$(dirname "$HC_NCCL_TESTS")"
    git clone --depth 1 --branch "$HC_NCCL_TESTS_TAG" https://github.com/NVIDIA/nccl-tests.git "$HC_NCCL_TESTS"
    make -C "$HC_NCCL_TESTS" -j 8 MPI=1 MPI_HOME="$EBROOTOPENMPI" \
        CUDA_HOME="${CUDA_HOME:-$EBROOTCUDACORE}" NCCL_HOME="$EBROOTNCCL" NVCC_GENCODE="$HC_NVCC_GENCODE"
fi

echo "== venv $HC_VENV"
mkdir -p "$(dirname "$HC_VENV")"
export UV_PROJECT_ENVIRONMENT="$HC_VENV" UV_CACHE_DIR="$HC_BASE/.uv-cache" UV_PYTHON_DOWNLOADS=never
cd "$HC_CODE"
unset VIRTUAL_ENV  # set by `uv run healthcheck.py`; this syncs $HC_VENV instead
flock "$HC_VENV.lock" "$HC_UV" sync --frozen --no-dev --python "$(command -v python3)"

echo "== read-test files"
"$HC_VENV/bin/python" -m healthcheck.scratch
echo "== build done"
