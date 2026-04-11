#!/usr/bin/env bash
# Create the `prosst` conda env with the pinned deps from the upstream repo.
# Safe to rerun — skips if env already exists.
set -euo pipefail

ENV_NAME="${1:-prosst}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROSST_REQ="${REPO_ROOT}/../ProSST/requirements.txt"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found on PATH" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "env '${ENV_NAME}' already exists — activating"
else
    echo "creating env '${ENV_NAME}' (python=3.10)"
    conda create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"

echo "installing torch 2.1.1 + cu118 (required by ProSST's torch-geometric wheels)"
pip install --upgrade pip
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

echo "installing torch-geometric stack (cu118 wheels)"
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.1+cu118.html
pip install torch_geometric==2.5.0

echo "installing ProSST pinned requirements (tolerating existing torch)"
# Use --no-deps where needed; the full req file pins everything.
pip install -r "${PROSST_REQ}" || {
    echo "full requirements install failed — retrying without torch pins"
    grep -vE '^(torch|torchvision|torchaudio|pyg|torch_scatter|torch_sparse|torch_cluster|torch_spline_conv|torch_geometric)' \
        "${PROSST_REQ}" > /tmp/prosst_req_notorch.txt
    pip install -r /tmp/prosst_req_notorch.txt
}

echo "sanity check"
python - <<'PY'
import torch, transformers, Bio, scipy, pandas
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("transformers", transformers.__version__)
PY

echo "done. Activate with: conda activate ${ENV_NAME}"
