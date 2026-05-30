"""End-to-end driver: protein + ligand similarity over the 10 train + 17 val.

Writes:
  similarity/meta_27.csv
  similarity/protein_emb_27.npz
  similarity/ligand_emb_27.npz
  similarity/per_complex_similarity.csv
  similarity/similarity_map.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.embed_ligands import embed_ligands
from src.data.embed_proteins import cosine_similarity_matrix, embed_complexes
from src.data.joint_similarity import (
    build_summary,
    kmeans_clusters,
    plot_similarity_map,
    umap_project,
)
from src.data.sequence_extract import load_directory


def main(train_dir: str, val_dir: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train = load_directory(train_dir)
    val = load_directory(val_dir)
    complexes = train + val
    sets = ["train"] * len(train) + ["val"] * len(val)
    pdb_ids = [c.pdb_id for c in complexes]
    print(f"loaded {len(train)} train + {len(val)} val complexes")

    # Meta
    meta_rows = []
    for c, s in zip(complexes, sets):
        lig = c.primary_ligand
        meta_rows.append(
            {
                "pdb_id": c.pdb_id,
                "set": s,
                "ligand_ccd": lig.ccd_id if lig else "",
                "ligand_smiles": (lig.smiles if (lig and lig.smiles) else ""),
                "seq_md5": c.seqres_md5,
                "n_chains": c.n_chains,
                "n_pocket_residues": sum(len(v) for v in c.pocket_residues.values()),
            }
        )
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(out / "meta_27.csv", index=False)
    print(f"wrote {out/'meta_27.csv'}")

    # Protein embeddings (whole + pocket)
    pe = embed_complexes(complexes)
    np.savez(out / "protein_emb_27.npz",
             pdb_ids=np.asarray(pe.pdb_ids), whole=pe.whole, pocket=pe.pocket)
    print(f"wrote {out/'protein_emb_27.npz'} (whole={pe.whole.shape}, pocket={pe.pocket.shape})")

    # Ligand embeddings
    smiles = [c.primary_ligand.smiles if c.primary_ligand else None for c in complexes]
    le = embed_ligands(pdb_ids, smiles)
    np.savez(out / "ligand_emb_27.npz",
             pdb_ids=np.asarray(pdb_ids),
             smiles=np.asarray([s or "" for s in smiles]),
             sim_matrix=le.sim_matrix,
             cluster_ids=le.cluster_ids)
    print(f"wrote {out/'ligand_emb_27.npz'}")

    # Joint similarity table
    protein_sim = cosine_similarity_matrix(pe.pocket)
    pocket_clusters = kmeans_clusters(pe.pocket, n_clusters=4)
    train_indices = [i for i, s in enumerate(sets) if s == "train"]
    df = build_summary(
        pdb_ids=pdb_ids,
        sets=sets,
        ligand_ccds=[m["ligand_ccd"] for m in meta_rows],
        ligand_smiles=[m["ligand_smiles"] for m in meta_rows],
        seq_md5s=[m["seq_md5"] for m in meta_rows],
        pocket_clusters=pocket_clusters,
        ligand_clusters=le.cluster_ids,
        protein_sim=protein_sim,
        ligand_sim=le.sim_matrix,
        train_indices=train_indices,
    )
    df.to_csv(out / "per_complex_similarity.csv", index=False)
    print(f"wrote {out/'per_complex_similarity.csv'} ({len(df)} rows)")

    # UMAP plot
    coords = umap_project(pe.pocket)
    plot_similarity_map(coords, pdb_ids, sets, le.cluster_ids, out / "similarity_map.png")
    print(f"wrote {out/'similarity_map.png'}")

    # Quick sanity printout
    n_lig_clusters = int((df["ligand_cluster"][df["ligand_cluster"] >= 0]).nunique())
    train_clusters = set(df[df["set"] == "train"]["ligand_cluster"])
    val_clusters = set(df[df["set"] == "val"]["ligand_cluster"])
    val_only = val_clusters - train_clusters
    print("\n=== sanity ===")
    # Tanimoto distribution (off-diagonal upper triangle)
    n = le.sim_matrix.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    tani = le.sim_matrix[iu, ju]
    print(f"pairwise Tanimoto: n={len(tani)}, mean={tani.mean():.3f}, "
          f"p25={np.quantile(tani,0.25):.3f}, p50={np.quantile(tani,0.5):.3f}, "
          f"p75={np.quantile(tani,0.75):.3f}, p95={np.quantile(tani,0.95):.3f}")
    print(f"ligand clusters detected: {n_lig_clusters}")
    print(f"train ligand clusters: {sorted(train_clusters)}")
    print(f"val   ligand clusters: {sorted(val_clusters)}")
    print(f"val-only clusters (uncovered by train): {sorted(val_only)}")
    print(f"mean joint sim of val to train: {df[df['set']=='val']['joint_similarity_score'].mean():.3f}")
    print(f"protein sequence MD5 groups: {meta['seq_md5'].value_counts().to_dict()}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1], sys.argv[2], sys.argv[3])
