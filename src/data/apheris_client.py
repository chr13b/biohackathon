"""Thin client around the Apheris Hub REST API on http://localhost:8080.

Three flows we need:
  * Poll a fine-tune job to completion and retrieve its metrics.
  * Deploy a fine-tuned weight under a chosen version string.
  * Run inference for each of N CIF+a3m complexes using either the base
    or the fine-tuned weights, and fetch per-sample metrics.

Each CIF+a3m pair is parsed locally (gemmi + rdkit) to build the JSON
queries.chains payload, since the inference endpoint requires the chain
spec — unlike fine-tune dataset uploads, which auto-parse the CIF on the
server.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests

API = "http://localhost:8080/api/v1"


@dataclass
class FineTuneSnapshot:
    job_id: str
    status: str
    started_at: str | None
    ended_at: str | None
    training_time_seconds: int
    base_metrics: dict | None
    hyper_params: dict
    dataset_sizes: dict
    raw: dict


def fetch_fine_tune(job_id: str) -> FineTuneSnapshot:
    r = requests.get(f"{API}/fine-tune/{job_id}", timeout=30)
    r.raise_for_status()
    d = r.json()
    return FineTuneSnapshot(
        job_id=d["id"],
        status=d.get("status"),
        started_at=d.get("startedAt"),
        ended_at=d.get("endedAt"),
        training_time_seconds=d.get("trainingTimeSeconds", 0),
        base_metrics=d.get("baseMetrics"),
        hyper_params=d.get("hyperParams", {}),
        dataset_sizes=d.get("datasetSizes", {}),
        raw=d,
    )


def poll_fine_tune(
    job_id: str,
    interval_s: int = 30,
    timeout_s: int = 60 * 60 * 12,
    on_update=None,
) -> FineTuneSnapshot:
    deadline = time.time() + timeout_s
    last_status = None
    while time.time() < deadline:
        snap = fetch_fine_tune(job_id)
        if snap.status != last_status:
            last_status = snap.status
            if on_update:
                on_update(snap)
        if snap.status in ("completed", "failed", "cancelled"):
            return snap
        time.sleep(interval_s)
    raise TimeoutError(f"fine-tune {job_id} did not finish in {timeout_s}s")


def deploy_fine_tune(job_id: str, version: str) -> dict:
    """Deploy the fine-tuned weights so they're usable for inference."""
    payload = {"version": version}
    r = requests.post(f"{API}/fine-tune/{job_id}/deploy", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


# --- Inference -------------------------------------------------------------


def _unwrap(j: dict) -> dict:
    """Apheris wraps most responses in {'data': ...}. Inputs/requests do, fine-tune doesn't."""
    return j.get("data", j)


def create_input(tag: str, queries: dict) -> str:
    """POST /api/v1/inputs to create a new inference input shell.

    `queries` is the requestParams.queries dict: {<tag>: {chains: [...]}}.
    Returns the input id.
    """
    payload = {"tags": [tag], "requestParams": {"queries": queries}}
    r = requests.post(f"{API}/inputs", json=payload, timeout=30)
    r.raise_for_status()
    return _unwrap(r.json())["id"]


def upload_input_asset(input_id: str, key: str, content_bytes: bytes) -> None:
    """PUT a file (e.g. the MSA) under the input's assets directory."""
    r = requests.put(
        f"{API}/inputs/{input_id}/assets/{key}",
        data=content_bytes,
        headers={"Content-Type": "application/octet-stream"},
        timeout=120,
    )
    r.raise_for_status()


def submit_inference(
    input_id: str,
    weight_version: str,
    model_version_id: str = "openfold3:0.60.0",
    diffusion_samples: int = 5,
    seed: int = 42,
) -> str:
    payload = {
        "inputId": input_id,
        "versionId": model_version_id,
        "weightVersion": weight_version,
        "capability": "inference",
        "modelParams": {"diffusion_samples": diffusion_samples, "seed": seed},
    }
    r = requests.post(f"{API}/requests", json=payload, timeout=30)
    r.raise_for_status()
    return _unwrap(r.json())["id"]


def poll_request(request_id: str, interval_s: int = 10, timeout_s: int = 60 * 30) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = requests.get(f"{API}/requests/{request_id}", timeout=30)
        r.raise_for_status()
        d = _unwrap(r.json())
        if d.get("status") in ("done", "failed", "cancelled"):
            return d
        time.sleep(interval_s)
    raise TimeoutError(f"request {request_id} did not finish")


def fetch_sample_meta(request_id: str, query_tag: str, sample_idx: int) -> dict:
    """Get the metadata JSON for a specific diffusion sample (has the LDDT scores)."""
    r = requests.get(
        f"{API}/requests/{request_id}/output/meta/{query_tag}_sample_{sample_idx}.json",
        timeout=30,
    )
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    return r.json()


def fetch_prediction_cif(request_id: str, query_tag: str, sample_idx: int) -> bytes:
    """Get the predicted CIF bytes for one sample."""
    r = requests.get(
        f"{API}/requests/{request_id}/predictions/{query_tag}/samples/{sample_idx}",
        timeout=60,
    )
    r.raise_for_status()
    return r.content


# --- Compose chains from a local CIF + a3m ---------------------------------


# The canonical Apheris PDE10A catalytic-domain MSA query sequence —
# bit-identical first row of every one of our 27 Apheris-supplied a3ms.
# Inference must use this sequence (matching the MSA's first row) not the
# observed CIF chain (which is shorter due to disordered termini).
CANONICAL_PDE10A_QUERY = (
    "GSSICTSEEWQGLMQFTLPVRLCKEIELFHFDIGPFENMWPGIFVYMVHRSCGTSCFELEKLCRFIMSVKKNYRRVPYHN"
    "WKHAVTVAHCMYAILQNNHTLFTDLERKGLLIACLCHDLDHRGFSNSYLQKFDHPLAALYSTSTMEQHHFSQTVSILQLE"
    "GHNIFSTLSSSEYEQVLEIIRKAIIATDLALYFGNRKQLEEMYQTGSLNLNNQSHRDRVIGLMMTACDLCSVTKLWPVTK"
    "LTANDIYAEFWAEGDEMKKLGIQPIPMMDRDKKDEVPQGQLGFYNAVAIPCYTTLTQILPPTEPLLKACRDNLSQWEKVI"
    "RGEETATWISSPSVAQKAAASED"
)


def cif_to_inference_queries(cif_path: Path, a3m_path: Path, query_tag: str) -> tuple[dict, str]:
    """Build the queries dict + the MSA filename to upload.

    For PDE10A inference we use the CANONICAL Apheris query sequence
    (which matches the a3m's first row exactly), NOT the observed CIF
    chain sequence (which is shorter due to disordered termini, and
    therefore wouldn't align to the MSA correctly).

    The ligand specification uses the primary non-solvent ligand's CCD
    code from the CIF.
    """
    from src.data.sequence_extract import load_complex

    comp = load_complex(cif_path)
    msa_filename = f"msa_{query_tag}_1.a3m"

    chains: list[dict] = [{
        "chain_ids": ["1"],
        "molecule_type": "protein",
        "msa": msa_filename,
        "sequence": CANONICAL_PDE10A_QUERY,
    }]
    next_id = 2
    if comp.primary_ligand:
        chains.append({
            "chain_ids": [str(next_id)],
            "molecule_type": "ligand",
            "ccd_codes": comp.primary_ligand.ccd_id,
        })
        next_id += 1

    queries = {query_tag: {"chains": chains}}
    return queries, msa_filename
