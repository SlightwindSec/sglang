#!/bin/bash
set -e

# 获取脚本所在目录（即 sglang 仓库根目录）
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${REPO_ROOT}/sglang_minicpm_sala_env"
DEPS_DIR="${REPO_ROOT}/3rdparty"

# PyPI mirror: prefer CLI argument, then env var, default to official source
if [ -n "$1" ]; then
    export UV_INDEX_URL="$1"
elif [ -z "${UV_INDEX_URL}" ]; then
    export UV_INDEX_URL="https://pypi.org/simple"
fi

echo "============================================"
echo " MiniCPM-SALA Installation (uv)"
echo "============================================"
echo "Root Directory: ${REPO_ROOT}"
echo "PyPI mirror:    ${UV_INDEX_URL}"

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Error: 'uv' is not installed. Please install it first (e.g., pip install uv)."
    exit 1
fi

# ---- Prepare Dependencies ----
mkdir -p "${DEPS_DIR}"

# 1. infllmv2_cuda_impl
if [ ! -d "${DEPS_DIR}/infllmv2_cuda_impl" ]; then
    echo "[0/4] Cloning infllmv2_cuda_impl..."
    git clone -b minicpm_sala https://github.com/OpenBMB/infllmv2_cuda_impl.git "${DEPS_DIR}/infllmv2_cuda_impl"
else
    echo "[0/4] infllmv2_cuda_impl already exists, skipping"
fi

# 2. sparse_kernel
# (Managed as git submodule)
if [ ! -d "${DEPS_DIR}/sparse_kernel/.git" ]; then
    echo "[0/4] Initializing sparse_kernel submodule..."
    git submodule update --init --recursive "${DEPS_DIR}/sparse_kernel"
fi

# ---- Create venv ----
if [ -d "${VENV_DIR}" ]; then
    VENV_PY_VER=$("${VENV_DIR}/bin/python" --version 2>&1 | awk '{print $2}')
    if [[ "${VENV_PY_VER}" != 3.12.* ]]; then
        echo "[1/4] venv exists but Python version is ${VENV_PY_VER} (expected 3.12.x), recreating..."
        rm -rf "${VENV_DIR}"
        uv venv --python 3.12 "${VENV_DIR}"
    else
        echo "[1/4] venv already exists (Python ${VENV_PY_VER}), skipping"
    fi
else
    echo "[1/4] Creating virtual environment (Python 3.12)..."
    uv venv --python 3.12 "${VENV_DIR}"
fi

# Activate environment variables for the script execution
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:$PATH"
echo "Python: $(python --version)"

# ---- Prepare build environment ----
# Fix compiler (python-build-standalone sets CXX="clang++ -pthread", incompatible with CMake)
if command -v g++ &> /dev/null; then
    export CC=gcc CXX=g++
fi
# Ensure nvcc is in PATH
if [ -z "${CUDA_HOME}" ]; then
    if [ -x /usr/local/cuda/bin/nvcc ]; then
        export CUDA_HOME="/usr/local/cuda"
    fi
fi
if [ -n "${CUDA_HOME}" ]; then
    export PATH="${CUDA_HOME}/bin:$PATH"
    export CUDACXX="${CUDA_HOME}/bin/nvcc"
fi

# ---- Install Packages ----

# 1. Install sglang (Current Directory)
echo "[2/4] Installing sglang (current directory)..."
uv pip install "cmake>=3.26"
uv pip install --upgrade pip setuptools wheel
# 安装当前目录 (sglang)
uv pip install -e "${REPO_ROOT}/python[all]"

# 2. Install Dependencies
echo "[3/4] Building CUDA kernels..."

# infllm_v2
echo "  - Installing infllm_v2..."
cd "${DEPS_DIR}/infllmv2_cuda_impl"
# 即使之前 clone 过，也确保 submodule 是最新的
git submodule update --init --recursive
python setup.py install

# sparse_kernel
echo "  - Installing sparse_kernel..."
cd "${DEPS_DIR}/sparse_kernel"
python setup.py install

# 3. Install Additional Libraries
echo "[4/4] Installing additional libraries..."
uv pip install tilelang flash-linear-attention

# ---- Verify ----
echo ""
echo "============================================"
echo " Installation complete!"
echo "============================================"
echo "To activate the environment, run:"
echo "source ${VENV_DIR}/bin/activate"
