"""Filter the 359 RCSB Q9Y233 hits, score by similarity, pick ~20-30 spanning
chemotype clusters, prioritising clusters our 27 don't cover.

Writes:
  similarity/augmentation_candidates.csv  # the full ranked list (filtered)
  similarity/augmentation_picks.csv       # the chosen ~20-30 with rationale

Does NOT download CIFs or assemble files yet — that's the next step
(`assemble_augmentation_set.py`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

# --- canonical Apheris MSA query sequence (the catalytic domain) -----------
# All 27 Apheris-provided a3ms (md5 8569ccaf...) share this query.
# An augmentation candidate is MSA-reuse-compatible iff this query appears
# as a substring (or near-substring) of its full polymer sequence.
CANONICAL_QUERY = (
    "GSSICTSEEWQGLMQFTLPVRLCKEIELFHFDIGPFENMWPGIFVYMVHRSCGTSCFELEKLCRFIMSVKKNYRRVPYHN"
    "WKHAVTVAHCMYAILQNNHTLFTDLERKGLLIACLCHDLDHRGFSNSYLQKFDHPLAALYSTSTMEQHHFSQTVSILQLE"
    "GHNIFSTLSSSEYEQVLEIIRKAIIATDLALYFGNRKQLEEMYQTGSLNLNNQSHRDRVIGLMMTACDLCSVTKLWPVTK"
    "LTANDIYAEFWAEGDEMKKLGIQPIPMMDRDKKDEVPQGQLGFYNAVAIPCYTTLTQILPPTEPLLKACRDNLSQWEKVI"
    "RGEETATWISSPSVAQKAAASED"
)
# Also keep the alt-MSA query for reference (used by 5shr_alt1).
ALT_QUERY_MD5_FALLBACK = True  # unused for now; placeholder for future logic

MORGAN_RADIUS = 2
MORGAN_BITS = 2048


# Our 10 training and 17 validation PDB IDs (from CLAUDE.md, lowercase).
TRAIN_IDS = {"5sdy", "5siq", "5si7", "5sig", "5si5", "5si8", "5siy", "5sg5", "5sgl", "5sih"}
VAL_IDS = {"5sh0", "5se0", "5shr", "5sjl", "5sh8", "5sf4", "5sfg", "5se5",
           "5shk", "5see", "5sfl", "5sju", "5ske", "5sku", "5sko", "5sea", "5skr"}
OURS = TRAIN_IDS | VAL_IDS


@dataclass
class FilteredCandidate:
    pdb_id: str
    ligand_ccd: str
    ligand_smiles: str
    ligand_mw: float
    resolution: float | None
    release_year: int | None
    msa_reuse_ok: bool
    msa_match_method: str   # "exact_substring" | "near_substring" | "no_match"
    seq_len: int


# ---------- Step 1: filter ----------------------------------------------------


def _sequence_compat(candidate_seq: str | None) -> tuple[bool, str]:
    if not candidate_seq:
        return False, "no_seq"
    # Strip any whitespace / line breaks.
    s = "".join(candidate_seq.split()).upper()
    if CANONICAL_QUERY in s:
        return True, "exact_substring"
    # Try without the leading "GSS" (expression tag) — equivalent for MSA purposes.
    if CANONICAL_QUERY[3:] in s:
        return True, "exact_substring_no_tag"
    # Near-substring: align the canonical query window-wise, allow up to 5 mismatches.
    # Cheap and fast for ~360 candidates.
    Q = CANONICAL_QUERY
    LQ = len(Q)
    if len(s) < LQ - 5:
        return False, "too_short"
    for offset in range(0, len(s) - LQ + 6):
        window = s[offset : offset + LQ]
        if len(window) < LQ - 5:
            break
        mismatches = sum(1 for a, b in zip(window, Q) if a != b)
        if mismatches <= 3:
            return True, f"near_substring(m={mismatches},off={offset})"
    return False, "no_match"


def filter_candidates(rows: list[dict]) -> list[FilteredCandidate]:
    """Apply hard filters: must have ligand SMILES, must be non-ours, must
    have an MSA-reuse-compatible sequence."""
    out: list[FilteredCandidate] = []
    for r in rows:
        pid = r["pdb_id"].lower()
        if pid in OURS:
            continue
        if not r.get("ligand_smiles") or not r.get("ligand_ccd"):
            continue
        ok, method = _sequence_compat(r.get("protein_sequence"))
        if not ok:
            continue
        rel = r.get("initial_release_date") or ""
        year = int(rel[:4]) if rel[:4].isdigit() else None
        out.append(
            FilteredCandidate(
                pdb_id=pid,
                ligand_ccd=r["ligand_ccd"],
                ligand_smiles=r["ligand_smiles"],
                ligand_mw=r.get("ligand_mw") or 0.0,
                resolution=r.get("resolution"),
                release_year=year,
                msa_reuse_ok=ok,
                msa_match_method=method,
                seq_len=len(r["protein_sequence"]),
            )
        )
    return out


# ---------- Step 2: similarity to our 27 + cluster ----------------------------


def _morgan(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)


def score_against_27(
    candidates: list[FilteredCandidate],
    ours_meta: pd.DataFrame,    # similarity/meta_27.csv with 'set' and 'ligand_smiles'
) -> pd.DataFrame:
    """For each candidate, compute max Tanimoto vs our train ligands and vs
    all 27 ligands. Returns a dataframe with one row per candidate."""
    train_smiles = [s for s, st in zip(ours_meta["ligand_smiles"], ours_meta["set"])
                    if st == "train" and s]
    val_smiles = [s for s, st in zip(ours_meta["ligand_smiles"], ours_meta["set"])
                  if st == "val" and s]
    train_fps = [_morgan(s) for s in train_smiles if _morgan(s) is not None]
    val_fps = [_morgan(s) for s in val_smiles if _morgan(s) is not None]
    train_fps = [fp for fp in train_fps if fp is not None]
    val_fps = [fp for fp in val_fps if fp is not None]

    rows = []
    for c in candidates:
        fp = _morgan(c.ligand_smiles)
        if fp is None:
            tani_train = tani_val = 0.0
            best_t = best_v = ""
        else:
            sims_t = DataStructs.BulkTanimotoSimilarity(fp, train_fps) if train_fps else [0.0]
            sims_v = DataStructs.BulkTanimotoSimilarity(fp, val_fps) if val_fps else [0.0]
            tani_train = float(max(sims_t))
            tani_val = float(max(sims_v))
            train_ids = [pid for pid, st in zip(ours_meta["pdb_id"], ours_meta["set"]) if st == "train"]
            val_ids = [pid for pid, st in zip(ours_meta["pdb_id"], ours_meta["set"]) if st == "val"]
            best_t = train_ids[int(np.argmax(sims_t))] if train_fps else ""
            best_v = val_ids[int(np.argmax(sims_v))] if val_fps else ""
        rows.append({
            "pdb_id": c.pdb_id,
            "ligand_ccd": c.ligand_ccd,
            "ligand_smiles": c.ligand_smiles,
            "ligand_mw": c.ligand_mw,
            "resolution": c.resolution,
            "release_year": c.release_year,
            "msa_reuse_ok": c.msa_reuse_ok,
            "msa_match_method": c.msa_match_method,
            "tanimoto_max_train": tani_train,
            "nearest_train_ligand": best_t,
            "tanimoto_max_val": tani_val,
            "nearest_val_ligand": best_v,
        })
    return pd.DataFrame(rows)


def cluster_candidates(df: pd.DataFrame, cutoff: float = 0.65) -> pd.DataFrame:
    """Butina-cluster candidate ligands together (with our 27 as anchors)
    so cluster ids are compatible across sets. Returns df with `cand_cluster`."""
    fps = [_morgan(s) for s in df["ligand_smiles"]]
    valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
    if not valid_idx:
        df["cand_cluster"] = -1
        return df
    valid_fps = [fps[i] for i in valid_idx]
    dists = []
    for i in range(1, len(valid_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(valid_fps[i], valid_fps[:i])
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(dists, len(valid_fps), cutoff, isDistData=True)
    cluster_ids = -np.ones(len(fps), dtype=int)
    for cid, members in enumerate(clusters):
        for m in members:
            cluster_ids[valid_idx[m]] = cid
    df = df.copy()
    df["cand_cluster"] = cluster_ids
    return df


# ---------- Step 3: pick the augmentation set --------------------------------


def pick_augmentation(
    df: pd.DataFrame,
    per_complex: pd.DataFrame,    # similarity/per_complex_similarity.csv (the 27)
    target_n: int = 25,
) -> pd.DataFrame:
    """Pick a chemotype-balanced set of ~target_n candidates.

    Heuristic:
      1. Identify the val-only ligand clusters in `per_complex` (those uncovered by train).
      2. For each such cluster, find candidate ligands whose nearest_val_ligand is in
         that cluster and tanimoto_max_val is high. Take 2-4 per uncovered cluster.
      3. Fill the rest with candidates that maximise diversity in `cand_cluster`
         space (sample 1-2 per cand_cluster, prioritising clusters underrepresented
         in our 27).
      4. Cap at target_n total.
    """
    # 1. uncovered val clusters
    train_clusters = set(per_complex.query("set=='train'")["ligand_cluster"])
    val_only = sorted(set(per_complex.query("set=='val'")["ligand_cluster"]) - train_clusters)
    val_only = [c for c in val_only if c >= 0]

    picks: list[pd.Series] = []
    used_ids: set[str] = set()

    # Resolve uncovered-val PDB ids
    uncovered_val_ids = set(
        per_complex.query("set=='val' and ligand_cluster in @val_only")["pdb_id"]
    )

    for c in val_only:
        target_val_ids = set(per_complex.query("set=='val' and ligand_cluster==@c")["pdb_id"])
        # candidates whose nearest val ligand is in this cluster
        bucket = df[df["nearest_val_ligand"].isin(target_val_ids)].copy()
        bucket = bucket.sort_values(["tanimoto_max_val", "resolution"],
                                    ascending=[False, True]).head(3)
        for _, row in bucket.iterrows():
            if row["pdb_id"] not in used_ids:
                used_ids.add(row["pdb_id"])
                row = row.copy()
                row["pick_reason"] = f"covers_val_cluster_{c}"
                picks.append(row)

    # 2. fill with chemotype diversity from cand_cluster space
    candidates_by_cluster = (
        df[~df["pdb_id"].isin(used_ids)]
        .sort_values(["resolution", "tanimoto_max_train"], ascending=[True, False])
        .groupby("cand_cluster")
        .head(1)
    )
    for _, row in candidates_by_cluster.iterrows():
        if len(picks) >= target_n:
            break
        if row["pdb_id"] in used_ids:
            continue
        used_ids.add(row["pdb_id"])
        row = row.copy()
        row["pick_reason"] = f"cluster_diversity_cluster_{int(row['cand_cluster'])}"
        picks.append(row)

    # 3. if still short, top up with high-tanimoto-to-val (covers near-uncovered cases)
    if len(picks) < target_n:
        remaining = (
            df[~df["pdb_id"].isin(used_ids)]
            .sort_values(["tanimoto_max_val"], ascending=False)
            .head(target_n - len(picks))
        )
        for _, row in remaining.iterrows():
            row = row.copy()
            row["pick_reason"] = "high_tanimoto_to_val"
            picks.append(row)

    out = pd.DataFrame(picks).reset_index(drop=True)
    return out


# ---------- main -------------------------------------------------------------


def main(
    metadata_json: Path,
    meta_27_csv: Path,
    per_complex_csv: Path,
    out_dir: Path,
    target_n: int = 25,
) -> None:
    rows = json.loads(metadata_json.read_text())
    print(f"loaded {len(rows)} RCSB metadata rows")

    candidates = filter_candidates(rows)
    print(f"after filter (ligand + dedup + MSA-compat): {len(candidates)} candidates")
    methods = {}
    for c in candidates:
        methods[c.msa_match_method] = methods.get(c.msa_match_method, 0) + 1
    print(f"MSA-match method distribution: {methods}")

    ours_meta = pd.read_csv(meta_27_csv)
    per_complex = pd.read_csv(per_complex_csv)

    scored = score_against_27(candidates, ours_meta)
    scored = cluster_candidates(scored)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_dir / "augmentation_candidates.csv", index=False)
    print(f"wrote {out_dir/'augmentation_candidates.csv'} ({len(scored)} rows)")

    picks = pick_augmentation(scored, per_complex, target_n=target_n)
    picks.to_csv(out_dir / "augmentation_picks.csv", index=False)
    print(f"wrote {out_dir/'augmentation_picks.csv'} ({len(picks)} picks)")

    # Summary
    reasons = picks["pick_reason"].value_counts().to_dict()
    print("\n=== pick reasons ===")
    for reason, count in reasons.items():
        print(f"  {reason}: {count}")
    print(f"\nmean tanimoto_max_train across picks: {picks['tanimoto_max_train'].mean():.3f}")
    print(f"mean tanimoto_max_val   across picks: {picks['tanimoto_max_val'].mean():.3f}")
    print(f"release-year distribution: {picks['release_year'].value_counts().sort_index().to_dict()}")
    print(f"resolution mean: {picks['resolution'].mean():.2f} A")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--meta-27", required=True)
    parser.add_argument("--per-complex", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--target-n", type=int, default=25)
    args = parser.parse_args()

    main(Path(args.metadata), Path(args.meta_27),
         Path(args.per_complex), Path(args.out_dir), args.target_n)
