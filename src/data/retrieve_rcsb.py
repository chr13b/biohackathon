"""Retrieve all RCSB PDB entries matching UniProt Q9Y233 (human PDE10A).

Two-phase pipeline:
  1. Search API → list of PDB IDs (359 at time of writing).
  2. GraphQL data API → polymer sequence + non-polymer ligand info + metadata
     for every ID. Persist to JSON.
  3. (Separate downloader) For chosen candidates, fetch the actual mmCIF.

Cache everything under ~/biohackathon-work/cache/rcsb/ so re-runs are cheap.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import requests


SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL_URL = "https://data.rcsb.org/graphql"
CIF_URL = "https://files.rcsb.org/download/{id}.cif"
DEFAULT_UNIPROT = "Q9Y233"  # human PDE10A
CACHE_DIR_DEFAULT = Path.home() / "biohackathon-work" / "cache" / "rcsb"


def search_uniprot(uniprot_id: str = DEFAULT_UNIPROT, max_rows: int = 1000) -> list[str]:
    """Full-text search for the UniProt ID, return PDB entry IDs."""
    payload = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {"value": uniprot_id},
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": max_rows},
            "results_content_type": ["experimental"],
        },
    }
    r = requests.post(SEARCH_URL, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    ids = [hit["identifier"] for hit in data.get("result_set", [])]
    print(f"[rcsb] search returned {len(ids)} entries for {uniprot_id} "
          f"(total_count={data.get('total_count')})")
    return ids


_GRAPHQL_QUERY = """
query ($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    rcsb_entry_info {
      deposited_polymer_monomer_count
      polymer_entity_count_protein
      nonpolymer_entity_count
      resolution_combined
    }
    rcsb_accession_info {
      deposit_date
      initial_release_date
    }
    polymer_entities {
      rcsb_polymer_entity_container_identifiers { entity_id }
      entity_poly { pdbx_seq_one_letter_code_can rcsb_entity_polymer_type }
      rcsb_polymer_entity_container_identifiers { auth_asym_ids }
    }
    nonpolymer_entities {
      rcsb_nonpolymer_entity_container_identifiers { entity_id }
      nonpolymer_comp {
        chem_comp { id name type formula_weight }
        rcsb_chem_comp_descriptor { SMILES InChI }
      }
    }
  }
}
"""


def fetch_metadata(ids: list[str], chunk: int = 50, sleep_s: float = 0.3) -> list[dict]:
    """GraphQL query in chunks; returns per-entry dicts."""
    out: list[dict] = []
    for i in range(0, len(ids), chunk):
        batch = ids[i : i + chunk]
        payload = {"query": _GRAPHQL_QUERY, "variables": {"ids": batch}}
        r = requests.post(GRAPHQL_URL, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        entries = (data.get("data") or {}).get("entries") or []
        for e in entries:
            if e is not None:
                out.append(e)
        if sleep_s:
            time.sleep(sleep_s)
        print(f"[rcsb] fetched metadata chunk {i//chunk+1}: cumulative {len(out)}")
    return out


def _extract_protein_sequence(entry: dict) -> tuple[str | None, str | None]:
    """Return (sequence, entity_id). Pick the first protein polymer entity."""
    for poly in entry.get("polymer_entities") or []:
        ep = poly.get("entity_poly") or {}
        if (ep.get("rcsb_entity_polymer_type") or "").lower() == "protein":
            seq = ep.get("pdbx_seq_one_letter_code_can")
            ent_id = (poly.get("rcsb_polymer_entity_container_identifiers") or {}).get("entity_id")
            if seq:
                return seq.replace("\n", "").strip(), ent_id
    return None, None


def _extract_primary_ligand(entry: dict) -> dict | None:
    """Return CCD + SMILES + name for the largest non-solvent ligand."""
    best = None
    best_mw = -1.0
    SOLVENT_CCD = {
        "HOH", "DOD", "EDO", "GOL", "PEG", "PG4", "MPD", "DMS", "SO4", "PO4",
        "NA", "K", "MG", "CA", "ZN", "FE", "MN", "CL", "ACT", "FMT", "TRS",
        "EPE", "MES", "BTB", "IOD", "CD", "NI", "CO", "CU", "HG", "BR", "F",
    }
    for nonp in entry.get("nonpolymer_entities") or []:
        comp = nonp.get("nonpolymer_comp") or {}
        cc = comp.get("chem_comp") or {}
        ccd = (cc.get("id") or "").upper()
        if not ccd or ccd in SOLVENT_CCD:
            continue
        descr = comp.get("rcsb_chem_comp_descriptor") or {}
        smiles = descr.get("SMILES") or descr.get("InChI") or None
        mw = float(cc.get("formula_weight") or 0)
        if mw > best_mw:
            best_mw = mw
            best = {
                "ccd": ccd,
                "name": cc.get("name"),
                "type": cc.get("type"),
                "smiles": smiles,
                "mw": mw,
            }
    return best


def normalize(metadata: list[dict]) -> list[dict]:
    """Flatten the GraphQL response into a per-entry dict ready for filtering."""
    out: list[dict] = []
    for e in metadata:
        pid = (e.get("rcsb_id") or "").upper()
        info = e.get("rcsb_entry_info") or {}
        acc = e.get("rcsb_accession_info") or {}
        seq, entity_id = _extract_protein_sequence(e)
        lig = _extract_primary_ligand(e)
        out.append({
            "pdb_id": pid,
            "deposit_date": acc.get("deposit_date"),
            "initial_release_date": acc.get("initial_release_date"),
            "resolution": (info.get("resolution_combined") or [None])[0]
                          if info.get("resolution_combined") else None,
            "nonpolymer_count": info.get("nonpolymer_entity_count"),
            "polymer_count_protein": info.get("polymer_entity_count_protein"),
            "n_residues": info.get("deposited_polymer_monomer_count"),
            "protein_sequence": seq,
            "protein_entity_id": entity_id,
            "ligand_ccd": lig.get("ccd") if lig else None,
            "ligand_smiles": lig.get("smiles") if lig else None,
            "ligand_name": lig.get("name") if lig else None,
            "ligand_mw": lig.get("mw") if lig else None,
        })
    return out


def download_cif(pdb_id: str, dest_dir: Path) -> Path:
    """Fetch the mmCIF for a single entry. Cached on disk."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{pdb_id.lower()}.cif"
    if out.exists() and out.stat().st_size > 1000:
        return out
    r = requests.get(CIF_URL.format(id=pdb_id), timeout=60)
    r.raise_for_status()
    out.write_text(r.text)
    return out


def cache_paths(cache_dir: Path = CACHE_DIR_DEFAULT) -> dict[str, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        "ids": cache_dir / "Q9Y233_ids.json",
        "metadata": cache_dir / "Q9Y233_metadata.json",
        "normalized": cache_dir / "Q9Y233_normalized.json",
        "cifs": cache_dir / "cifs",
    }


def main(
    uniprot_id: str = DEFAULT_UNIPROT,
    cache_dir: Path = CACHE_DIR_DEFAULT,
    refresh: bool = False,
) -> list[dict]:
    paths = cache_paths(cache_dir)

    if paths["ids"].exists() and not refresh:
        ids = json.loads(paths["ids"].read_text())
        print(f"[rcsb] loaded cached id list ({len(ids)} entries)")
    else:
        ids = search_uniprot(uniprot_id)
        paths["ids"].write_text(json.dumps(ids))

    if paths["normalized"].exists() and not refresh:
        normalized = json.loads(paths["normalized"].read_text())
        print(f"[rcsb] loaded cached normalized metadata ({len(normalized)} entries)")
    else:
        metadata = fetch_metadata(ids)
        paths["metadata"].write_text(json.dumps(metadata))
        normalized = normalize(metadata)
        paths["normalized"].write_text(json.dumps(normalized, indent=2))

    return normalized


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--uniprot", default=DEFAULT_UNIPROT)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    rows = main(args.uniprot, refresh=args.refresh)
    print(f"\n=== summary of {len(rows)} entries ===")
    with_ligand = [r for r in rows if r["ligand_ccd"]]
    with_smiles = [r for r in rows if r["ligand_smiles"]]
    with_seq = [r for r in rows if r["protein_sequence"]]
    print(f"with primary ligand: {len(with_ligand)}")
    print(f"with ligand SMILES:  {len(with_smiles)}")
    print(f"with protein seq:    {len(with_seq)}")
    if rows:
        res = [r["resolution"] for r in rows if r["resolution"]]
        if res:
            print(f"resolution range:   {min(res):.2f} - {max(res):.2f} A")
        dates = sorted([r["initial_release_date"][:4] for r in rows if r["initial_release_date"]])
        if dates:
            from collections import Counter
            print(f"release year distribution: {dict(Counter(dates))}")
