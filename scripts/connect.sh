#!/usr/bin/env bash
# Open an interactive SSH session on the team VM.
# Usage: bash scripts/connect.sh
# Requires: $VM_IP exported in your shell; team-4 key at ~/.ssh/team-4.
set -euo pipefail

: "${VM_IP:?VM_IP not set — export VM_IP=<ip> in your shell first}"
KEY="${HOME}/.ssh/team-4"
[[ -f "$KEY" ]] || { echo "Missing key: $KEY" >&2; exit 1; }

ssh -i "$KEY" "lyceum@${VM_IP}"
