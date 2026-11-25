# Run by `healthcheck.py setup` on the login node, since compute nodes may have no internet:
# nccl-tests, the shared venv and $SCRATCH and $PROJECT read-test files.
set -euo pipefail

# Lmod; an exported module() may lack its helper. Trillium's /etc/profile reads unset variables.
module --version &>/dev/null || { set +u; source /etc/profile; set -u; }
HC_ENV="$HC_CODE/slurm/scripts/.$HC_CLUSTER.env"  # a cluster's own, else the Alliance clusters' shared one
[[ -n $HC_CLUSTER && -f $HC_ENV ]] || HC_ENV="$HC_CODE/slurm/scripts/.drac.env"
source "$HC_ENV"
if [[ -n $HC_MODULES ]]; then module load $HC_MODULES; fi  # Lynx has no Lmod
HC_PYTHON=$("$HC_UV" python find --project "$HC_CODE")  # .python-version's, from a module or uv's own installs

if [[ -n ${HC_MPI_PIP:-} ]]; then  # clusters whose MPI doesn't fit their Slurm
    echo "== mpi $HC_MPI_PIP"
    if [[ ! -e $HC_MPI_HOME/lib/libmpi.so ]]; then
        "$HC_UV" pip install --quiet --python "$HC_PYTHON" --target "$HC_MPI_HOME" "$HC_MPI_PIP"
    fi
fi

if [[ -n ${HC_CUDA_PIP:-} ]]; then  # clusters without a CUDA toolkit
    echo "== cuda $HC_CUDA_PIP"
    if [[ ! -x $CUDA_HOME/bin/nvcc ]]; then
        "$HC_UV" pip install --quiet --python "$HC_PYTHON" --target "$HC_CUDA_TARGET" $HC_CUDA_PIP
        ln -sfn lib "$CUDA_HOME/lib64"  # where nccl-tests looks
        ln -sf libcudart.so.13 "$CUDA_HOME/lib/libcudart.so"
    fi
fi

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
