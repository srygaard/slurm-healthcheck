# Run by `healthcheck.py setup` on the login node, since compute nodes may have no internet:
# nccl-tests, the shared venv and $SCRATCH and $PROJECT read-test files.
set -euo pipefail

type module &>/dev/null || source /etc/profile  # Lmod
if [[ $HC_CLUSTER == mila ]]; then source "$HC_CODE/slurm/scripts/.mila.env"; else source "$HC_CODE/slurm/scripts/.drac.env"; fi
module load $HC_MODULES
HC_PYTHON=$("$HC_UV" python find --project "$HC_CODE")  # .python-version's, from a module or uv's own installs

if [[ -n ${HC_NCCL_PIP:-} ]]; then  # clusters without an NCCL module
    echo "== nccl $HC_NCCL_PIP"
    if [[ ! -e $HC_NCCL_HOME/lib/libnccl.so ]]; then
        "$HC_UV" pip install --quiet --python "$HC_PYTHON" --target "$HC_BASE/.opt/nccl" "$HC_NCCL_PIP"
        ln -sf libnccl.so.2 "$HC_NCCL_HOME/lib/libnccl.so"
    fi
fi

echo "== nccl-tests $HC_NCCL_TESTS_TAG"
if [[ ! -x $HC_NCCL_TESTS/build/all_reduce_perf ]]; then
    rm -rf "$HC_NCCL_TESTS"
    mkdir -p "$(dirname "$HC_NCCL_TESTS")"
    git clone --depth 1 --branch "$HC_NCCL_TESTS_TAG" https://github.com/NVIDIA/nccl-tests.git "$HC_NCCL_TESTS"
    make -C "$HC_NCCL_TESTS" -j 8 MPI=1 MPI_HOME="${HC_MPI_HOME:-$EBROOTOPENMPI}" \
        CUDA_HOME="${CUDA_HOME:-$EBROOTCUDACORE}" NCCL_HOME="${HC_NCCL_HOME:-$EBROOTNCCL}" NVCC_GENCODE="$HC_NVCC_GENCODE"
fi

echo "== venv $HC_VENV"
mkdir -p "$(dirname "$HC_VENV")"
export UV_PROJECT_ENVIRONMENT="$HC_VENV" UV_CACHE_DIR="$HC_BASE/.uv-cache" UV_PYTHON_DOWNLOADS=never
cd "$HC_CODE"
unset VIRTUAL_ENV  # set by `uv run healthcheck.py`; this syncs $HC_VENV instead
flock "$HC_VENV.lock" "$HC_UV" sync --frozen --no-dev --python "$HC_PYTHON"

echo "== read-test files"
"$HC_VENV/bin/python" -m healthcheck.share
echo "== build done"
