#!/usr/bin/env bash
# Upload a fine-tuned OF3 checkpoint to the VM and register it with ApherisFold.
# Usage: bash scripts/deploy_weights.sh <local_checkpoint.pt> <version_string>
# Example: bash scripts/deploy_weights.sh ./of3_d0a.pt 3.0.0-d0a-short
set -euo pipefail

: "${VM_IP:?VM_IP not set — export VM_IP=<ip> in your shell first}"
KEY="${HOME}/.ssh/team-4"
[[ -f "$KEY" ]] || { echo "Missing key: $KEY" >&2; exit 1; }

CKPT="${1:?usage: deploy_weights.sh <checkpoint.pt> <version_string>}"
VER="${2:?usage: deploy_weights.sh <checkpoint.pt> <version_string>}"
[[ -f "$CKPT" ]] || { echo "Checkpoint not found: $CKPT" >&2; exit 1; }

REMOTE_DIR="weights_mount/fine-tuned"
REMOTE_NAME="of3_${VER//./_}.pt"   # safe filename derived from version

echo "[1/4] scp -> ${REMOTE_DIR}/${REMOTE_NAME}"
ssh -i "$KEY" "lyceum@${VM_IP}" "mkdir -p ${REMOTE_DIR}"
scp -i "$KEY" "$CKPT" "lyceum@${VM_IP}:${REMOTE_DIR}/${REMOTE_NAME}"

echo "[2/4] registering version='${VER}' in additional_weights.json"
ssh -i "$KEY" "lyceum@${VM_IP}" \
  "VER='${VER}' REMOTE_NAME='${REMOTE_NAME}' bash -s" <<'REMOTE'
set -euo pipefail
cd weights_mount
test -f additional_weights.json || echo '{"available_weights":[]}' > additional_weights.json
python3 - <<PY
import json, pathlib
p = pathlib.Path("additional_weights.json")
cfg = json.loads(p.read_text() or '{"available_weights":[]}')
cfg.setdefault("available_weights", [])
ver = "${VER}"
entry = {
    "model_type": "openfold3",
    "version": ver,
    "description": f"PDE10A fine-tuned weights ({ver})",
    "model_scope": ["inference"],
    "mounted_path": "/weights/openfold3/fine-tuned",
}
cfg["available_weights"] = [w for w in cfg["available_weights"] if w.get("version") != ver]
cfg["available_weights"].append(entry)
p.write_text(json.dumps(cfg, indent=2) + "\n")
print("registered:", ver)
PY
REMOTE

echo "[3/4] ./deploy_apherisfold"
ssh -i "$KEY" "lyceum@${VM_IP}" "./deploy_apherisfold"

echo "[4/4] ./deploy_apherisfold diagnose"
ssh -i "$KEY" "lyceum@${VM_IP}" "./deploy_apherisfold diagnose"

echo
echo "Done. Reload the Hub UI; '${VER}' should appear in the Predict-page dropdown."
