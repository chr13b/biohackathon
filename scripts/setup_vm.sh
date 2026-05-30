#!/usr/bin/env bash
# First-time bootstrap on the VM: unzip the dataset and sanity-check it.
# Usage (run ON the VM after SSH): bash setup_vm.sh
set -euo pipefail

ZIP=/apheris/apherisfold_inputs.zip
OUT=/apheris/inputs

if [[ ! -f "$ZIP" ]]; then
  echo "Expected dataset at $ZIP — not found." >&2
  exit 1
fi

mkdir -p "$OUT"
if [[ -z "$(ls -A "$OUT" 2>/dev/null || true)" ]]; then
  echo "Unzipping $ZIP -> $OUT"
  unzip -q "$ZIP" -d "$OUT"
else
  echo "$OUT already populated — skipping unzip."
fi

echo
echo "--- Top-level contents ---"
ls "$OUT" | head -50

# Expected PDB IDs from CLAUDE.md.
TRAIN=(5SDY 5SIQ 5SI7 5SIG 5SI5 5SI8 5SIY 5SG5 5SGL 5SIH)
EVAL=(5SH0 5SE0 5SHR 5SJL 5SH8 5SF4 5SFG 5SE5 5SHK 5SEE 5SFL 5SJU 5SKE 5SKU 5SKO 5SEA 5SKR)

echo
echo "--- CIF/a3m presence check (case-insensitive) ---"
missing=0
for id in "${TRAIN[@]}" "${EVAL[@]}"; do
  cif=$(find "$OUT" -iname "${id}.cif" -print -quit)
  a3m=$(find "$OUT" -iname "${id}*.a3m" -print -quit)
  flag=""
  [[ -z "$cif" ]] && { flag+=" cif_MISSING"; missing=$((missing+1)); }
  [[ -z "$a3m" ]] && { flag+=" a3m_MISSING"; missing=$((missing+1)); }
  [[ -n "$flag" ]] && printf "  %s%s\n" "$id" "$flag"
done

if [[ "$missing" -eq 0 ]]; then
  echo "  All 27 IDs have CIF + a3m present."
else
  echo "  $missing missing file(s) — investigate before training." >&2
fi

echo
echo "--- GPU ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || echo "nvidia-smi not available"

echo
echo "Bootstrap complete. Hub UI: open scripts/tunnel.sh on your laptop."
