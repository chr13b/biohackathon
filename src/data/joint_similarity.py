"""Joint protein × ligand similarity + UMAP/PCA projection.

Joint score for a (query, candidate) pair = mean of protein-pocket cosine
similarity and ligand Tanimoto. Symmetric, in [0, 1].

For each complex we also compute:
- nearest_train_protein_sim: best protein-pocket sim to any complex in the
  train pool (excluding the complex itself if it's in train).
- nearest_train_ligand_tanimoto: same for the ligand side.
- joint_similarity_score: mean of the two above.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import umap  # umap-learn
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False


@dataclass
class JointSimilarity:
    pdb_ids: list[str]
    protein_sim: np.ndarray
    ligand_sim: np.ndarray
    joint_sim: np.ndarray


def joint_similarity(protein_sim: np.ndarray, ligand_sim: np.ndarray) -> np.ndarray:
    """Element-wise mean of protein and ligand similarity matrices."""
    assert protein_sim.shape == ligand_sim.shape
    return (protein_sim + ligand_sim) / 2.0


def build_summary(
    pdb_ids: list[str],
    sets: list[str],
    ligand_ccds: list[str | None],
    ligand_smiles: list[str | None],
    seq_md5s: list[str],
    pocket_clusters: np.ndarray,
    ligand_clusters: np.ndarray,
    protein_sim: np.ndarray,
    ligand_sim: np.ndarray,
    train_indices: list[int],
) -> pd.DataFrame:
    """Build per-complex similarity table including nearest-train fields."""
    rows = []
    for i, pid in enumerate(pdb_ids):
        is_self_in_train = i in train_indices
        train_pool = [k for k in train_indices if not (is_self_in_train and k == i)]
        if train_pool:
            j_p = int(np.argmax(protein_sim[i, train_pool]))
            best_p_idx = train_pool[j_p]
            best_p_sim = float(protein_sim[i, best_p_idx])
            j_l = int(np.argmax(ligand_sim[i, train_pool]))
            best_l_idx = train_pool[j_l]
            best_l_sim = float(ligand_sim[i, best_l_idx])
        else:
            best_p_idx = best_l_idx = -1
            best_p_sim = best_l_sim = float("nan")
        joint = (best_p_sim + best_l_sim) / 2 if best_p_idx >= 0 else float("nan")
        rows.append({
            "pdb_id": pid,
            "set": sets[i],
            "ligand_ccd": ligand_ccds[i],
            "ligand_smiles": ligand_smiles[i] or "",
            "protein_seq_md5": seq_md5s[i],
            "esm_pocket_cluster": int(pocket_clusters[i]) if pocket_clusters[i] >= 0 else -1,
            "ligand_cluster": int(ligand_clusters[i]) if ligand_clusters[i] >= 0 else -1,
            "nearest_train_protein_id": pdb_ids[best_p_idx] if best_p_idx >= 0 else "",
            "nearest_train_protein_sim": best_p_sim,
            "nearest_train_ligand_id": pdb_ids[best_l_idx] if best_l_idx >= 0 else "",
            "nearest_train_ligand_tanimoto": best_l_sim,
            "joint_similarity_score": joint,
        })
    return pd.DataFrame(rows)


def kmeans_clusters(embeddings: np.ndarray, n_clusters: int = 4, seed: int = 42) -> np.ndarray:
    """Simple cluster id assignment for the protein-side embeddings.
    Using K-means with a small k since all 27 proteins are PDE10A and the
    signal we want is sub-cluster (construct boundaries / pocket variation).
    """
    from sklearn.cluster import KMeans

    n = embeddings.shape[0]
    k = max(2, min(n_clusters, n - 1))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    return km.fit_predict(embeddings)


def umap_project(embeddings: np.ndarray, seed: int = 42) -> np.ndarray:
    """2-D UMAP. Falls back to PCA if umap-learn missing."""
    n = embeddings.shape[0]
    if _HAS_UMAP and n >= 4:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, max(2, n - 1)),
            min_dist=0.1,
            metric="cosine",
            random_state=seed,
        )
        return reducer.fit_transform(embeddings)
    from sklearn.decomposition import PCA

    return PCA(n_components=2, random_state=seed).fit_transform(embeddings)


def plot_similarity_map(
    coords: np.ndarray,
    pdb_ids: list[str],
    sets: list[str],
    ligand_clusters: np.ndarray,
    out_path: Path,
) -> None:
    """Static PNG: UMAP coords coloured by ligand cluster, marker = set."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    markers = {"train": "s", "val": "o", "candidate": "+"}
    unique_clusters = sorted({int(c) for c in ligand_clusters if c >= 0})
    cmap = plt.get_cmap("tab20")
    legend_seen: set[str] = set()
    for i, pid in enumerate(pdb_ids):
        s = sets[i]
        m = markers.get(s, ".")
        c_idx = int(ligand_clusters[i])
        color = "#cccccc" if c_idx < 0 else cmap(unique_clusters.index(c_idx) % 20)
        label = f"{s} cluster {c_idx}" if (s, c_idx) not in legend_seen else None
        if label:
            legend_seen.add((s, c_idx))
        ax.scatter(coords[i, 0], coords[i, 1], marker=m, c=[color], s=80,
                   edgecolors="black", linewidths=0.4, label=label)
        ax.annotate(pid, (coords[i, 0], coords[i, 1]),
                    fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title("Protein pocket ESM-2 embedding; marker = set; colour = ligand cluster")
    ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--protein-emb", required=True, help=".npz from embed_proteins")
    parser.add_argument("--ligand-emb", required=True, help=".npz from embed_ligands")
    parser.add_argument("--meta", required=True,
                        help="CSV with columns: pdb_id, set, ligand_ccd, seq_md5")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-plot", required=True)
    args = parser.parse_args()

    pe = np.load(args.protein_emb, allow_pickle=True)
    le = np.load(args.ligand_emb, allow_pickle=True)
    meta = pd.read_csv(args.meta)
    # Align rows by pdb_id
    pdb_ids = list(pe["pdb_ids"])
    assert list(le["pdb_ids"]) == pdb_ids, "ligand vs protein emb pdb_id order mismatch"
    meta = meta.set_index("pdb_id").reindex(pdb_ids).reset_index()

    from src.data.embed_proteins import cosine_similarity_matrix

    protein_sim = cosine_similarity_matrix(pe["pocket"])
    ligand_sim = le["sim_matrix"]
    pocket_clusters = kmeans_clusters(pe["pocket"], n_clusters=4)
    ligand_clusters = le["cluster_ids"]

    train_indices = [i for i, s in enumerate(meta["set"]) if s == "train"]

    df = build_summary(
        pdb_ids=pdb_ids,
        sets=list(meta["set"]),
        ligand_ccds=list(meta["ligand_ccd"]),
        ligand_smiles=list(le["smiles"]) if "smiles" in le.files else [None] * len(pdb_ids),
        seq_md5s=list(meta["seq_md5"]),
        pocket_clusters=pocket_clusters,
        ligand_clusters=ligand_clusters,
        protein_sim=protein_sim,
        ligand_sim=ligand_sim,
        train_indices=train_indices,
    )
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[joint_similarity] wrote {args.out_csv} ({len(df)} rows)")

    coords = umap_project(pe["pocket"])
    plot_similarity_map(coords, pdb_ids, list(meta["set"]), ligand_clusters, Path(args.out_plot))
    print(f"[joint_similarity] wrote {args.out_plot}")
