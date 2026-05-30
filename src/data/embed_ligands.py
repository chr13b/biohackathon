"""RDKit Morgan-r2-2048 fingerprints + Tanimoto similarity + Butina clustering."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina


MORGAN_RADIUS = 2
MORGAN_BITS = 2048
# Butina groups pairs with Tanimoto distance < cutoff. For fragment-sized
# ligands, average pairwise Tanimoto is often 0.15-0.30, so a cutoff of 0.5
# (= similarity > 0.5) puts almost every fragment in its own micro-cluster.
# 0.65 (= similarity > 0.35) is the chemotype-scaffold regime relevant to
# us; tune at the CLI if needed.
BUTINA_CUTOFF = 0.65


@dataclass
class LigandEmbeddings:
    pdb_ids: list[str]
    smiles: list[str | None]
    fingerprints: list           # list[ExplicitBitVect] for RDKit Tanimoto calls
    sim_matrix: np.ndarray       # (N, N) Tanimoto
    cluster_ids: np.ndarray      # (N,) cluster index (-1 = no usable SMILES)


def _morgan_fp(smiles: str | None):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=MORGAN_BITS)


def _butina_cluster(fps: list, cutoff: float = BUTINA_CUTOFF) -> np.ndarray:
    """Returns per-fingerprint cluster id; None-fp rows get -1."""
    valid_idx = [i for i, fp in enumerate(fps) if fp is not None]
    if not valid_idx:
        return -np.ones(len(fps), dtype=int)
    valid_fps = [fps[i] for i in valid_idx]
    # Build flat distance vector as Butina expects.
    dists: list[float] = []
    for i in range(1, len(valid_fps)):
        sims = DataStructs.BulkTanimotoSimilarity(valid_fps[i], valid_fps[:i])
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(dists, len(valid_fps), cutoff, isDistData=True)
    cluster_ids = -np.ones(len(fps), dtype=int)
    for cid, members in enumerate(clusters):
        for m in members:
            cluster_ids[valid_idx[m]] = cid
    return cluster_ids


def embed_ligands(pdb_ids: list[str], smiles: list[str | None]) -> LigandEmbeddings:
    fps = [_morgan_fp(s) for s in smiles]
    n = len(pdb_ids)
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if fps[i] is None:
            continue
        for j in range(i, n):
            if fps[j] is None:
                continue
            t = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sim[i, j] = t
            sim[j, i] = t
    clusters = _butina_cluster(fps)
    return LigandEmbeddings(
        pdb_ids=pdb_ids,
        smiles=smiles,
        fingerprints=fps,
        sim_matrix=sim,
        cluster_ids=clusters,
    )


def nearest_in(
    sim_row: np.ndarray,
    pool_indices: list[int],
    self_index: int | None = None,
) -> tuple[int, float]:
    """Return (best_pool_index, best_sim) excluding self_index if given."""
    best_idx = -1
    best_sim = -1.0
    for k in pool_indices:
        if k == self_index:
            continue
        if sim_row[k] > best_sim:
            best_sim = float(sim_row[k])
            best_idx = k
    return best_idx, best_sim


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from src.data.sequence_extract import load_directory

    parser = argparse.ArgumentParser()
    parser.add_argument("cif_dirs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    complexes: list = []
    for d in args.cif_dirs:
        complexes.extend(load_directory(d))
    smiles = [c.primary_ligand.smiles if c.primary_ligand else None for c in complexes]
    pdb_ids = [c.pdb_id for c in complexes]
    out = embed_ligands(pdb_ids, smiles)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        pdb_ids=np.asarray(pdb_ids),
        smiles=np.asarray([s or "" for s in smiles]),
        sim_matrix=out.sim_matrix,
        cluster_ids=out.cluster_ids,
    )
    n_valid = int(np.sum(out.cluster_ids >= 0))
    n_clusters = int(out.cluster_ids[out.cluster_ids >= 0].max() + 1) if n_valid > 0 else 0
    print(f"[embed_ligands] {n_valid}/{len(complexes)} with usable SMILES, {n_clusters} clusters")
