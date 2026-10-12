#!/usr/bin/env bash
set -euo pipefail

# Download pretrained ALERT checkpoints from a GitHub release and place
# them at the paths expected by the YAML configs in config/.
#
# Usage:
#   bash scripts/download_weights.sh           # defaults to tag v1.0
#   bash scripts/download_weights.sh v1.1      # override release tag
#
# Requires: curl. Idempotent: skips files that already exist with non-zero size.

TAG="${1:-v1.0}"
REPO="shormat/ALERT"
BASE="https://github.com/${REPO}/releases/download/${TAG}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

declare -A FILES=(
  ["yolov8m_bdd100k.pt.gz"]="checkpoints/detection/yolov8m_bdd100k/weights/best.pt"
  ["rtdetr-l_bdd100k.pt.gz"]="checkpoints/detection/rtdetr-l_bdd100k/weights/best.pt"
  ["segformer_bdd100k.pt.gz"]="checkpoints/segmentation/segformer_bdd100k/best.pt"
  ["deeplabv3plus_bdd100k.pt.gz"]="checkpoints/segmentation/deeplabv3plus_bdd100k/best.pt"
)

for tool in curl gunzip; do
  command -v "${tool}" >/dev/null 2>&1 || {
    echo "error: ${tool} not found in PATH"
    exit 1
  }
done

echo "Downloading ALERT weights from release ${TAG}..."
for asset in "${!FILES[@]}"; do
  dest="${FILES[$asset]}"
  if [[ -s "${dest}" ]]; then
    echo "  skip ${dest} (exists, $(du -h "${dest}" | cut -f1))"
    continue
  fi
  mkdir -p "$(dirname "${dest}")"
  url="${BASE}/${asset}"
  tmp="${dest}.gz"
  echo "  fetch ${asset} -> ${dest}"
  curl -fL --retry 3 --progress-bar -o "${tmp}" "${url}"
  gunzip -f "${tmp}"
done

echo
echo "Done. Verify with:"
echo "  ls -lh checkpoints/detection/*/weights/best.pt checkpoints/segmentation/*/best.pt"
