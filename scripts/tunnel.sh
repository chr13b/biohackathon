#!/usr/bin/env bash
# Forward the ApherisFold Hub UI to http://localhost:8081 and hold it open.
# Usage: bash scripts/tunnel.sh
# Requires: $VM_IP exported; team-4 key at ~/.ssh/team-4.
set -euo pipefail

: "${VM_IP:?VM_IP not set — export VM_IP=<ip> in your shell first}"
KEY="${HOME}/.ssh/team-4"
[[ -f "$KEY" ]] || { echo "Missing key: $KEY" >&2; exit 1; }

echo "Hub UI will be available at http://localhost:8081 — Ctrl-C to close."
ssh -i "$KEY" -N -L 8081:localhost:8080 "lyceum@${VM_IP}"
