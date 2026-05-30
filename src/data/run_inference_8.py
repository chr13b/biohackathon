"""Run inference for the 8 eval complexes using either base or FT weights,
and collect per-complex PL LDDT / IP LDDT / pose RMSD into a CSV.

Usage:
  python -m src.data.run_inference_8 \\
      --eval-dir /home/lyceum/biohackathon-work/dataset/eval \\
      --weight-version 4.0.0 \\
      --out results/inference/base_4.0.0.csv

For the FT side, --weight-version is the version string used in `deploy_fine_tune`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.data.apheris_client import (
    cif_to_inference_queries,
    create_input,
    fetch_sample_meta,
    poll_request,
    submit_inference,
    upload_input_asset,
)


def run_inference_for_one(
    cif_path: Path,
    a3m_path: Path,
    weight_version: str,
    diffusion_samples: int = 5,
) -> dict:
    """Submit + poll inference for a single complex; return its metrics."""
    qid = cif_path.stem
    queries, msa_filename = cif_to_inference_queries(cif_path, a3m_path, qid)

    input_id = create_input(qid, queries)
    upload_input_asset(input_id, msa_filename, a3m_path.read_bytes())

    request_id = submit_inference(
        input_id=input_id,
        weight_version=weight_version,
        diffusion_samples=diffusion_samples,
    )
    print(f"[infer] {qid}: input={input_id} request={request_id}")
    result = poll_request(request_id, interval_s=10, timeout_s=60 * 20)
    if result.get("status") != "done":
        return {"queryId": qid, "status": result.get("status"), "request_id": request_id}

    # Collect sample-level metrics (per diffusion sample)
    samples = []
    for s in range(diffusion_samples):
        meta = fetch_sample_meta(request_id, qid, s)
        if meta:
            samples.append({"sample": s, **meta})
    if not samples:
        return {"queryId": qid, "status": "no_meta", "request_id": request_id}

    df = pd.DataFrame(samples)
    # Try to extract the standard LDDT-PLI / IP LDDT / RMSD if present.
    # The meta JSON keys are model-version dependent; collect them all.
    metrics: dict = {"queryId": qid, "status": "done", "request_id": request_id,
                     "n_samples": len(samples)}
    for col in df.columns:
        if col in ("sample",):
            continue
        try:
            metrics[f"{col}_mean"] = float(df[col].mean())
            metrics[f"{col}_max"] = float(df[col].max())
        except Exception:
            pass
    # Always keep the raw sample dump on disk for offline reanalysis
    raw_dump_path = Path("results/inference") / f"raw_{weight_version.replace('.','_')}_{qid}.json"
    raw_dump_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dump_path.write_text(json.dumps({"request_id": request_id, "samples": samples}, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--weight-version", required=True,
                        help="Either '4.0.0' (base) or the version we deployed our FT weights as.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--diffusion-samples", type=int, default=5)
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    cifs = sorted(eval_dir.glob("*.cif"))
    print(f"[run_inference_8] found {len(cifs)} CIFs in {eval_dir}")

    rows = []
    for cif in cifs:
        a3m = cif.with_name(cif.stem + "_msa.a3m")
        if not a3m.exists():
            print(f"[run_inference_8] WARN: no a3m for {cif.name}, skipping")
            continue
        try:
            row = run_inference_for_one(cif, a3m, args.weight_version, args.diffusion_samples)
        except Exception as exc:
            row = {"queryId": cif.stem, "status": "exception", "error": str(exc)}
        rows.append(row)
        print(f"  -> {row}")

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[run_inference_8] wrote {args.out}")


if __name__ == "__main__":
    main()
