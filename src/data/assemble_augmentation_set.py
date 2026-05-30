"""Download CIFs for the picks and assemble the augmentation_set/ directory
with the canonical PDE10A MSA reused for each pick.

For each picked PDB ID we:
  1. Download <id>.cif from RCSB.
  2. Sanity-parse the CIF (load_complex) to confirm it has chains + a ligand.
     We do NOT re-check sequence compatibility here — the picker already
     verified the SEQRES (from RCSB GraphQL) contains the canonical query.
     The observed chain in the CIF is shorter (~310 residues vs 339 in the
     canonical query) because of disordered termini, which is normal — our
     own 27 a3ms have the same length disparity vs their chains.
  3. Copy the canonical a3m to <id>_msa.a3m.
  4. Write a manifest row with all the bookkeeping.

The resulting layout matches Apheris's training-file expectations:
  augmentation_set/<id>.cif
  augmentation_set/<id>_msa.a3m
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.data.retrieve_rcsb import download_cif
from src.data.sequence_extract import load_complex


def main(
    picks_csv: Path,
    canonical_a3m: Path,
    cif_cache: Path,
    out_dir: Path,
) -> None:
    picks = pd.read_csv(picks_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for _, row in picks.iterrows():
        pid = str(row["pdb_id"]).lower()
        try:
            cif_path = download_cif(pid.upper(), cif_cache)
        except Exception as exc:
            print(f"[assemble] {pid}: download failed -> {exc}")
            manifest_rows.append({**row.to_dict(), "status": "download_failed", "reason": str(exc)})
            continue

        try:
            comp = load_complex(cif_path)
        except Exception as exc:
            print(f"[assemble] {pid}: CIF parse failed -> {exc}")
            manifest_rows.append({**row.to_dict(), "status": "parse_failed", "reason": str(exc)})
            continue

        if not comp.primary_ligand:
            print(f"[assemble] {pid}: parsed CIF has no detectable ligand — SKIP")
            manifest_rows.append({**row.to_dict(), "status": "no_ligand_in_cif", "reason": ""})
            continue

        # Copy CIF + a3m into augmentation_set with the expected naming.
        dest_cif = out_dir / f"{pid}.cif"
        dest_a3m = out_dir / f"{pid}_msa.a3m"
        shutil.copyfile(cif_path, dest_cif)
        shutil.copyfile(canonical_a3m, dest_a3m)

        manifest_rows.append({
            **row.to_dict(),
            "status": "ok",
            "reason": "",
            "cif_bytes": dest_cif.stat().st_size,
            "a3m_bytes": dest_a3m.stat().st_size,
            "n_chains": comp.n_chains,
            "primary_ligand_ccd_in_cif": comp.primary_ligand.ccd_id,
            "n_pocket_residues": sum(len(v) for v in comp.pocket_residues.values()),
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    n_ok = int((manifest["status"] == "ok").sum())
    n_skip = int((manifest["status"] != "ok").sum())
    print(f"\n[assemble] wrote {manifest_path}")
    print(f"[assemble] {n_ok} OK / {n_skip} skipped")
    if n_skip > 0:
        skips = manifest[manifest["status"] != "ok"][["pdb_id", "status", "reason"]]
        print(skips.to_string(index=False))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--picks", required=True)
    parser.add_argument("--canonical-a3m", required=True,
                        help="Path to any of the existing 27 _msa.a3m files (all 27 are identical)")
    parser.add_argument("--cif-cache", required=True,
                        help="Directory for cached downloaded CIFs")
    parser.add_argument("--out-dir", required=True,
                        help="Final augmentation_set directory")
    args = parser.parse_args()

    main(Path(args.picks), Path(args.canonical_a3m),
         Path(args.cif_cache), Path(args.out_dir))
