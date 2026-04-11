#!/usr/bin/env bash
# Download the ProteinGym AlphaFold2 PDB dump hosted by the ProtSSN authors.
# ~3.5 GB zipped. Writes to ca_prosst/data/pdb/ by default.
set -euo pipefail

OUT_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/data/pdb}"
URL="https://huggingface.co/datasets/tyang816/ProteinGym_v1/resolve/main/ProteinGym_v1_AlphaFold2_PDB.zip"

mkdir -p "${OUT_DIR}"
cd "${OUT_DIR}"

if [[ -f ProteinGym_v1_AlphaFold2_PDB.zip ]]; then
    echo "zip already present at ${OUT_DIR} — skipping download"
else
    echo "downloading to ${OUT_DIR}"
    curl -L --fail -o ProteinGym_v1_AlphaFold2_PDB.zip "${URL}"
fi

echo "unpacking"
unzip -n ProteinGym_v1_AlphaFold2_PDB.zip
echo "done: $(find . -name '*.pdb' | wc -l) PDB files under ${OUT_DIR}"
