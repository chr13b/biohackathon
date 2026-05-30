"""Build split_proposal.json with several candidate train/eval partitions.

Splits emitted:

  - "original": the published 10-train / 17-val split.
  - "rebalanced": shrink eval to spanning held-out (one per discovered val
    ligand cluster + 1-2 furthest-from-train), move the rest into train,
    plus the 25 augmentation picks.
  - "leave_one_ligand_cluster_out": for each val ligand cluster, a slice
    that holds out only its members as eval (everything else into train).
    Used as a stricter generalisation probe at eval time.

Selection of the rebalanced eval relies on `per_complex_similarity.csv`
(produced by run_27.py) for ligand cluster ids and joint similarity scores.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _list_to_lower(xs) -> list[str]:
    return [str(x).lower() for x in xs]


def build_splits(
    per_complex_csv: Path,
    augmentation_manifest: Path,
    out_path: Path,
    held_out_per_cluster: int = 1,
    extra_far_holdouts: int = 1,
) -> dict:
    df = pd.read_csv(per_complex_csv)
    aug = pd.read_csv(augmentation_manifest)
    aug_ok_ids = _list_to_lower(aug[aug["status"] == "ok"]["pdb_id"])

    splits: dict = {}

    # ---- original (10 / 17) -------------------------------------------------
    original_train = _list_to_lower(df[df["set"] == "train"]["pdb_id"])
    original_val = _list_to_lower(df[df["set"] == "val"]["pdb_id"])
    splits["original"] = {
        "description": "Published Apheris split: 10 train / 17 val.",
        "train": sorted(original_train),
        "val": sorted(original_val),
    }

    # ---- rebalanced --------------------------------------------------------
    val_rows = df[df["set"] == "val"].copy()
    held_out: list[str] = []
    val_clusters = sorted(c for c in val_rows["ligand_cluster"].unique() if c >= 0)

    # one per val ligand cluster (lowest joint_similarity_score in that cluster -
    # i.e. the eval point most distinct from train, the toughest)
    for c in val_clusters:
        bucket = val_rows[val_rows["ligand_cluster"] == c].sort_values(
            "joint_similarity_score", ascending=True
        )
        for _, row in bucket.head(held_out_per_cluster).iterrows():
            held_out.append(str(row["pdb_id"]).lower())

    # +N additional held-outs from the very lowest joint_similarity_score not yet picked
    remaining_val = val_rows[~val_rows["pdb_id"].str.lower().isin(held_out)]
    extras = remaining_val.sort_values("joint_similarity_score", ascending=True).head(extra_far_holdouts)
    held_out.extend(_list_to_lower(extras["pdb_id"]))

    held_out = sorted(set(held_out))
    reassigned_val_to_train = sorted(set(original_val) - set(held_out))
    rebalanced_train = sorted(set(original_train) | set(reassigned_val_to_train) | set(aug_ok_ids))

    splits["rebalanced"] = {
        "description": (
            f"Rebalanced for the v2 FT run: {len(rebalanced_train)} train "
            f"(={len(original_train)} original + {len(reassigned_val_to_train)} moved from val + "
            f"{len(aug_ok_ids)} augmentation picks), {len(held_out)} held-out eval spanning "
            f"{len(val_clusters)} ligand clusters."
        ),
        "train": rebalanced_train,
        "eval_held_out": held_out,
        "reassigned_val_to_train": reassigned_val_to_train,
        "augmentation_added": sorted(aug_ok_ids),
    }

    # ---- leave-one-ligand-cluster-out evaluation slices --------------------
    locos: dict[str, dict] = {}
    for c in val_clusters:
        members = _list_to_lower(val_rows[val_rows["ligand_cluster"] == c]["pdb_id"])
        if not members:
            continue
        # train = original train + augmentation + all other val (everything except this cluster)
        eval_set = sorted(members)
        train_set = sorted(
            set(original_train)
            | set(aug_ok_ids)
            | (set(original_val) - set(members))
        )
        locos[f"cluster_{int(c)}"] = {
            "description": f"Leave-out: hold cluster {int(c)} ({len(eval_set)} members) as eval.",
            "train": train_set,
            "eval_held_out": eval_set,
        }
    splits["leave_one_ligand_cluster_out"] = locos

    # ---- ligand-cluster coverage report ------------------------------------
    # train_from_27: ligand clusters covered by entries that were in our 27
    # train_from_aug: clusters credited to augmentation via the picker's
    # `pick_reason` strings of the form "covers_val_cluster_<int>" or
    # "cluster_diversity_cluster_<int>". The latter is in cand_cluster
    # namespace (not directly comparable to the 27's ligand_cluster) so we
    # only credit "covers_val_cluster" entries here.
    import re
    train_27_clusters = set(
        int(c) for c in
        df[df["pdb_id"].str.lower().isin(rebalanced_train)]["ligand_cluster"]
        if c >= 0
    )
    aug_credited_clusters: set[int] = set()
    for _, row in aug.iterrows():
        if row.get("status") != "ok":
            continue
        m = re.match(r"covers_val_cluster_(\d+)", str(row.get("pick_reason") or ""))
        if m:
            aug_credited_clusters.add(int(m.group(1)))
    train_clusters_total = train_27_clusters | aug_credited_clusters

    val_clusters_after = (
        df[df["pdb_id"].str.lower().isin(held_out)]["ligand_cluster"]
    )
    eval_clusters = {int(c) for c in val_clusters_after if c >= 0}
    splits["coverage_report"] = {
        "train_27_ligand_clusters": sorted(train_27_clusters),
        "train_clusters_credited_to_augmentation": sorted(aug_credited_clusters),
        "train_clusters_total": sorted(train_clusters_total),
        "rebalanced_eval_ligand_clusters": sorted(eval_clusters),
        "uncovered_after_rebalance": sorted(eval_clusters - train_clusters_total),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(splits, indent=2))
    return splits


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--per-complex", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--held-out-per-cluster", type=int, default=1)
    parser.add_argument("--extra-far-holdouts", type=int, default=1)
    args = parser.parse_args()

    s = build_splits(
        Path(args.per_complex),
        Path(args.manifest),
        Path(args.out),
        held_out_per_cluster=args.held_out_per_cluster,
        extra_far_holdouts=args.extra_far_holdouts,
    )
    print("\n=== split summary ===")
    print(f"original: {len(s['original']['train'])} train / {len(s['original']['val'])} val")
    r = s["rebalanced"]
    print(f"rebalanced: {len(r['train'])} train / {len(r['eval_held_out'])} eval")
    print(f"  reassigned from val to train: {len(r['reassigned_val_to_train'])}")
    print(f"  augmentation added: {len(r['augmentation_added'])}")
    print(f"  eval IDs: {r['eval_held_out']}")
    print(f"locos: {len(s['leave_one_ligand_cluster_out'])} slices")
    cov = s["coverage_report"]
    print(f"\ncoverage: eval clusters {cov['rebalanced_eval_ligand_clusters']}, "
          f"uncovered after rebalance: {cov['uncovered_after_rebalance']}")
