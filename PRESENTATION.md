# Presentation walk-through — Team 4

> A slide-by-slide narrative for a general technical audience. Each `##`
> heading is one slide. Suggested visuals are called out in **italics**.
> A self-contained webpage rendering with the same content is in
> [`PRESENTATION.html`](./PRESENTATION.html). The long-form data-lane
> writeup is in [`html_files/report.html`](./html_files/report.html).

---

## Slide 1 — The question

> *Can a focused fine-tune on PDE10A — drawing on the 27 organiser-provided
> complexes plus structures from the public PDB — meaningfully improve
> OpenFold3's pose accuracy on held-out PDE10A complexes?*

**Answer: yes.** On 8 held-out complexes we improve mean Ligand-Protein
Interaction LDDT from **0.56 → 0.76** and cut mean Ligand RMSD from
**5.34 Å → 3.03 Å**, while protein Cα LDDT *rises* (0.964 → 0.989). The
result, in one bar chart, is on slide 9.

The target is **human PDE10A**, a phosphodiesterase that is a real
drug-discovery target with ~360 co-crystal structures in the PDB. We
have **24 hours**, **one A100-80GB**, and the public **OpenFold3 4.0.0**
weights. The organisers explicitly framed this as a **domain adaptation
problem**.

---

## Slide 2 — Why "change the data, not the hyperparameters"

A full fine-tune of OpenFold3 on this VM takes ~3 hours. In 24 hours
we get at most 2 attempts at FT, realistically 1.

A hyperparameter sweep needs ≥4 runs to be meaningful. A data
intervention needs **1 run**. The arithmetic is brutal: sweeping
hyperparameters under this budget produces noise; curating the
training set produces a signal.

So we asked a different question: **what 35 PDE10A complexes, if seen
during fine-tuning, would best teach OpenFold3 to model the eight
held-out ones?**

*Visual: a 2-row Gantt — top row "hyperparameter sweep (4 × 3 h = 12 h)",
bottom row "single FT + data curation (3 h)". Same wall-clock budget,
very different statistical power.*

---

## Slide 3 — What we knew at hour 0

The organisers handed us:
- **27 PDE10A co-crystal structures** (10 train + 17 val), all from the
  **Diamond XChem fragment screen** (PDB IDs start with `5S`).
- A canonical **MSA** for each — the protein sequence alignment that
  OpenFold3 uses as input.
- A working ApherisFold UI with one published FT recipe.

The first thing we did was **inspect the MSAs**. All 27 a3m files turned
out to be **bit-identical** (md5 `8569ccaf...`). One canonical MSA
covers every PDE10A entry.

This is the single most consequential discovery of the day. It means
**adding any other PDE10A PDB entry to training is free of MSA cost** —
just copy the canonical MSA. Without this, augmentation would have
required hours of MSA generation per new entry.

---

## Slide 4 — The provided split is adversarially hard

The 10/17 train/val split looks innocuous. It is not. When you cluster
the ligands by chemical similarity (Butina, Morgan-r2-2048 fingerprints,
0.65 cutoff), you find:

- **Train covers clusters** {0, 3, 4, 9–14}
- **Val covers clusters** {0, 1, 2, 5–8}
- **6 of 7 val clusters have no training counterpart at all.**

Random splits across these 27 would not measure generalisation —
they'd measure memorisation. The published split already enforces
across-chemotype generalisation. That makes the bar high.

*Visual: a 2-column bar chart of cluster occupancy in train vs val,
clusters on the x-axis, count on the y-axis, train and val side by side.
Or just the ligand-cluster heatmap from `results/similarity/`.*

---

## Slide 5 — Building a chemotype map

We embedded both sides of every complex:
- **Protein pocket** — sliced the 5 Å neighbourhood around the ligand,
  embedded the residues with **ESM-2 (35M, 12-layer)**, mean-pooled.
- **Ligand** — extracted SMILES, computed **Morgan-r2-2048** fingerprints,
  measured **Tanimoto similarity**.

Combined into a joint similarity score (mean of pocket-cosine and
ligand-Tanimoto), then projected onto a 2D map with **UMAP**.

This map is the basis for everything that follows: the augmentation
picks, the rebalanced split, and the held-out selection.

*Visual: `results/similarity/similarity_map.png` — the UMAP. Coloured
by ligand cluster, markers for train vs val vs augmentation pool.*

---

## Slide 6 — Mining the PDB

The RCSB returns **359 PDE10A entries** for UniProt Q9Y233. Filtering
for (a) a bound non-solvent ligand, (b) sequence compatibility with the
canonical MSA, and (c) deduplication against our 27, we're left with
**271 viable augmentation candidates**.

Each carries a joint similarity score to every one of our 27. We rank
them and **pick 16** — chosen to (i) cover val ligand clusters that the
original train doesn't, and (ii) avoid redundancy with the existing 10
train complexes.

Why only 16, not all 271? The Apheris reference recipe was tuned for
N≈10. Scaling to 271 changes the gradient-noise regime and may diverge
in our 10 h window. 16 is roughly 3× the original signal — broad enough
to matter, conservative enough to stay in the recipe's safe zone.

*Visual: a horizontal bar showing 359 RCSB hits → 271 viable → 16 picks,
with brief filter reasons stacked under each arrow.*

---

## Slide 7 — The actually-used dataset

**`dataset/train/` (35 entries)**
- 10 original Apheris train
- 9 reassigned from Apheris val (the more in-distribution ones)
- 16 augmentations from RCSB 5S\* fragment screen

**`dataset/eval/` (8 entries)**
- One per ligand cluster + 1 extra-far
- IDs: `5sh0, 5sh8, 5shk, 5shr, 5sju, 5ske, 5skr, 5sku`

All 43 entries share the canonical MSA. All ligands are real, deposited,
co-crystallised. **No synthetic poses, no docking.**

*Visual: a sankey-ish diagram from "Apheris-provided" / "RCSB" sources
on the left to "train (35) / eval (8)" on the right.*

---

## Slide 8 — Running the fine-tune

Config (the JSON pasted into the Hub UI — `configs/final_fine_tuned.json`):

```json
{
  "batch_size": 4,           "crop_size": 384,
  "learning_rate": 0.0003,   "warmup_steps": 50,
  "ema_decay": 0.99,         "precision": "bf16",
  "num_gradient_steps_per_epoch": 8,
  "save_top_k": -1,
  "data_seed": 42,           "training_seed": 42,
  "metric_to_monitor": "lddt_inter_protein_ligand",
  "maximum_training_time": 36000
}
```

`batch_size: 4` was the largest batch that stayed stable on the A100 at
`crop_size: 384` for our dataset — twice the throughput of the
published `batch_size: 1` reference.

The hub spawns one checkpoint every **8 gradient steps** (~5 min on the
A100). `save_top_k: -1` keeps **all** of them, so we have the full
PL LDDT curve.

The run launched cleanly, produced **35 checkpoints over ~3 hours**, then
crashed at gradient step 280 with `[Errno 28] No space left on device`
on the VM. We recovered the last checkpoint —
`checkpoint_of_first_run/34-280.ckpt`.

*Visual: a screenshot of the Hub UI Fine-tune progress view, or a
chronology bar showing the 3-hour run with the disk-exhaust event marked.*

---

## Slide 9 — What the FT achieved (the headline)

**Paired inference, base OF3 4.0.0 vs our FT, on the same 8 held-out
complexes, run minutes apart**:

| Metric (mean over 8) | Base | FT | Δ |
|---|---:|---:|---:|
| **Ligand-Protein Interaction LDDT** (↑) | **0.563** | **0.762** | **+0.200** |
| **Ligand RMSD** (Å, ↓) | **5.34** | **3.03** | **−2.31** |
| **Protein Cα LDDT** (↑, general-perf) | **0.964** | **0.989** | **+0.025** |
| **Protein Cα RMSD** (Å, ↓) | **1.69** | **1.40** | **−0.29** |

Per-complex (sorted by base PL LDDT):

| PDB | base PL | FT PL | Δ PL | base RMSD | FT RMSD | Δ RMSD |
|---|---:|---:|---:|---:|---:|---:|
| 5sh0 | 0.26 | 0.72 | **+0.46** | 8.63 | 2.97 | **−5.66** |
| 5sku | 0.28 | 0.28 | 0.00 | 10.41 | 10.44 | +0.03 |
| 5sh8 | 0.41 | 0.98 | **+0.57** | 7.27 | 0.33 | **−6.94** |
| 5ske | 0.52 | 0.92 | **+0.40** | 5.03 | 0.81 | **−4.22** |
| 5shk | 0.58 | 0.64 | +0.06 | 4.54 | 2.79 | −1.75 |
| 5shr | 0.64 | 0.69 | +0.05 | 5.24 | 5.29 | +0.05 |
| 5skr | 0.90 | 0.93 | +0.03 | 0.79 | 0.79 | 0.00 |
| 5sju | 0.91 | 0.94 | +0.03 | 0.81 | 0.84 | +0.03 |

Numbers live in `results/runs/standard_openfold_performance.csv` (base)
and `results/runs/prediction_fine_tuned.csv` (FT).

### Why the deltas look the way they do — a four-part typology

To explain the pattern we cross-referenced each held-out with its row in
`results/similarity/per_complex_similarity.csv` — specifically the
**Tanimoto similarity of its ligand to the nearest training ligand**
and the **Butina ligand cluster** it sits in. That gives a clean story:

| PDB | base PL | FT PL | base RMSD | FT RMSD | cluster | T(nearest-train-ligand) | regime |
|---|---:|---:|---:|---:|---:|---:|---|
| 5sh8 | 0.41 | 0.98 | 7.27 | 0.33 | 0 | 0.35 (vs `5sdy`) | **flip ↑** |
| 5sh0 | 0.26 | 0.72 | 8.63 | 2.97 | 8 | 0.22 (vs `5sg5`) | **flip ↑** |
| 5ske | 0.52 | 0.92 | 5.03 | 0.81 | 1 | 0.33 (vs `5sdy`) | **flip ↑** |
| 5shk | 0.58 | 0.64 | 4.54 | 2.79 | 7 | 0.17 (vs `5sih`) | partial |
| 5shr | 0.64 | 0.69 | 5.24 | 5.29 | 6 | 0.21 (vs `5sg5`) | partial |
| 5skr | 0.90 | 0.93 | 0.79 | 0.79 | 2 | 0.29 (vs `5si8`) | **ceiling** |
| 5sju | 0.91 | 0.94 | 0.81 | 0.84 | 0 | 0.35 (vs `5sdy`) | **ceiling** |
| 5sku | 0.28 | 0.28 | 10.41 | 10.44 | 5 | 0.23 (vs `5sg5`) | **floor** |

1. **Flip (5sh8, 5sh0, 5ske, Δ PL +0.40 to +0.57).** Base OF3 was
   misdocking these (RMSD 5–9 Å) but their chemotype neighbourhood
   *is* present in the FT training set — the augmentation lane
   specifically targeted clusters 0, 1, and 8 with the 5S\* picks.
   `5sh8` and `5ske` sit in clusters 0/1 which are anchored by
   `5sdy` (Tanimoto 0.33–0.35 to the held-out ligand). `5sh0` is in
   cluster 8 which the rebalanced split deliberately covered via
   augmentation. In all three cases, base is bad-but-not-catastrophic
   (RMSD < 9 Å), so once the FT learns the right pocket conformation
   it pulls the pose into the right basin.
2. **Ceiling (5skr, 5sju, Δ PL +0.03).** Both already had PL LDDT > 0.90
   and ligand RMSD < 1 Å on base OF3. PL LDDT saturates at 1.0; there
   is no room for an absolute gain like +0.4 on a 0.91 baseline. The
   relevant outcome here is **preservation**: focused fine-tuning
   often breaks formerly-easy cases via catastrophic forgetting, and
   ours didn't. Both stayed within 0.04 Å of their original pose.
3. **Partial (5shk, 5shr, Δ PL +0.05 to +0.06).** These ligands sit in
   clusters 6 and 7, both originally uncovered by the published 10
   train and only partly filled by our augmentation set. Their
   nearest-train ligand Tanimoto is the lowest in the panel
   (0.17–0.21) — the chemotype context is genuinely thin. We still see
   modest movement (5shk's RMSD drops from 4.54 → 2.79 Å, a 1.75 Å
   correction; 5shr is essentially unchanged), consistent with "we
   gave the model just enough chemotype exposure to nudge, not
   enough to flip." These two are the prime candidates for **targeted
   v2 augmentation** — more cluster-6/7 entries from the 271 viable
   RCSB pool we set aside.
4. **Floor (5sku, Δ PL 0.00).** The only case where FT made no
   difference at all. Its ligand
   (`CCOC1CC(F)CCC1C1NC(CCOC2CCC(CC(OC)C(O)O)C3CCCCC23)C(C)O1`) is
   the largest and most conformationally flexible in the eval panel —
   two fused cyclohexane systems linked by multi-atom ether chains.
   Base OF3 is 10.4 Å off — the prediction is in the wrong pose basin
   entirely. The nearest-train Tanimoto is 0.23, and cluster 5 was
   one of the more weakly covered clusters in our augmentation pass.
   The hypothesis: when (a) the chemotype is sparsely represented,
   (b) the ligand has many rotatable bonds, and (c) the base prediction
   is far off, a 35-complex FT doesn't carry enough signal to relocate
   the pose. Fixing this case probably needs either a dedicated
   augmentation pass focused on cluster 5 / large-flexible PDE10A
   inhibitors (the 2OU\*, 2Y0J, 3UI7, 3WS9 series we filtered out as
   MSA-incompatible) or a coarse-pose initialisation seed.

**The general rule the typology produces:** improvement is bounded
below by how lost base OF3 was *and* above by how well the held-out
chemotype is represented in the FT data. The intersection of those
two windows is where focused fine-tuning helps most — exactly where
our three biggest wins live.

- **General performance is preserved (and slightly better).** Protein
  Cα LDDT goes 0.96 → 0.99 — the FT did not damage the backbone
  prediction quality, the standard concern with focused fine-tuning.

*Visual: paired bar chart, one bar pair per held-out, sorted by base
PL LDDT ascending so the wins are visually obvious. PRESENTATION.html
renders this inline.*

---

## Slide 10 — What's real, what's curated, what we changed

For an outside reviewer who asks "did you just train on test data?":

1. The **eval set** (8 complexes) was held out from the very start. The
   FT job never saw them as training input — they only appeared as the
   Hub's "Validation Files" pile, which the FT uses for periodic
   read-only metric computation.
2. The **augmentation set** is sourced from the PDB. They are **other
   PDE10A structures, not the eval structures.** The maximum ligand
   Tanimoto from any augmentation pick to any eval ligand is **0.16** —
   well below the 0.65 cluster threshold.
3. We changed **which** PDE10A structures the model fine-tunes on, not
   the eval pile, not the loss, not the architecture.

So the experiment measures: *does broader PDE10A pocket exposure
during FT improve OpenFold3 on chemotypes it has not seen?* The
answer above (+0.20 PL LDDT on the held-out) is the measurement.

---

## Slide 11 — Recovering from disk exhaustion

The original run died at ~3 h with `[Errno 28]`. The VM's `/output`
volume filled because `save_top_k: -1` plus `num_gradient_steps_per_epoch: 8`
meant a checkpoint every ~5 minutes — and OpenFold3 checkpoints are large.

What we did:
1. Pulled the per-checkpoint metric JSON via the Apheris REST API —
   `GET /api/v1/fine-tune/<job_id>` returns the full `metrics` array
   with PL LDDT / IP LDDT per emitted checkpoint.
2. Copied the latest checkpoint (`34-280.ckpt`) off the VM before the
   container restarted.
3. Committed it to `checkpoint_of_first_run/` in this repo.
4. Re-ran paired base + FT inference on the 8 held-out from the
   recovered checkpoint — that produced `results/runs/*.csv`.

For a future re-run, the obvious fix is either `save_top_k: 3` (keep
the best 3 only) or to point `/output` at the larger `/data` volume.

---

## Slide 12 — Reproducibility

Anyone can reproduce this end-to-end. The repo is organised so the
mapping from "what we did" to "what's committed" is one-to-one:

| What we did | Lives at |
|---|---|
| The 35 train + 8 eval complexes uploaded to the Hub | `dataset/{train,eval}/` |
| The Settings JSON pasted into the Hub UI | `configs/final_fine_tuned.json` |
| The final checkpoint | `checkpoint_of_first_run/34-280.ckpt` |
| The base-OF3 numbers we compared against | `results/runs/standard_openfold_performance.csv` |
| Our FT inference numbers | `results/runs/prediction_fine_tuned.csv` |
| The similarity-lane pipeline that built the dataset | `src/data/*.py` |
| The exact shell + UI sequence | `scripts/README.md` |

Three paths from `README.md`:

- **(A) Skip training, just evaluate** — register the checkpoint with
  the Hub, run inference on the 8 eval, compare against the committed
  CSVs. ~30 min.
- **(B) Full FT from scratch** — upload `dataset/`, paste
  `configs/final_fine_tuned.json`, hit Start. ~3 h.
- **(C) Re-derive the dataset** — re-run the similarity lane
  (`src/data/*.py`). ~20 min on CPU.

The split, the config, the checkpoint, the inference code, the
similarity-lane pipeline, and even the disk-crash recovery story
are all committed.

---

## Slide 13 — Bottom line

- We treated this as a **domain-adaptation problem** and built a
  distribution-aware dataset rather than chasing hyperparameters.
- The **canonical-MSA discovery** made PDB-scale augmentation free.
- The **35-complex curated train set** improved held-out Ligand-Protein
  Interaction LDDT by **+0.20 absolute (35 % relative)** and cut
  Ligand RMSD by **2.31 Å (43 %)**, while protein Cα LDDT **also rose**
  — no general-performance regression.
- The training run died from disk exhaustion at 3 h, but the recovered
  checkpoint reproduces the gain and is committed.
- The whole pipeline — data, code, config, weights, results — is
  reproducible from this repo with three terminal commands.

*Closing visual: the per-complex bar chart from slide 9, with the
two flips (5sh8, 5sh0) highlighted.*

---

## Appendix — Suggested figures already in the repo

| Slide | Figure | Path |
|---|---|---|
| 5 | UMAP of pocket-ESM embeddings, coloured by ligand cluster | `results/similarity/similarity_map.png` |
| 7 | Per-complex provenance | derivable from `dataset/manifest.json` |
| 9 | Per-complex base-vs-FT bars | rendered inline in `PRESENTATION.html`; raw data in `results/runs/*.csv` |
| 13 | Same as 9 with the flips highlighted | same source |

A self-contained webpage rendering of this whole walk-through with
the headline chart drawn inline is in
[`PRESENTATION.html`](./PRESENTATION.html).
