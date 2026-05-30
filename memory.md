# memory.md — Living Project State

> Source of truth across sessions, teammates, and context compaction.
> **Read this first each session. Update it whenever something changes — and always
> flush open threads here before context gets compacted.**

_Last updated: 2026-05-30 (afternoon)_
_Updated by: Claude Code session 2 (Opus 4.7)_

---

## 1. Current status (one paragraph)

Plan v2 approved (domain-adaptation framing, not hyperparam tuning). VM
access verified; GPU is **A100-80GB** (not H100 as docs say), idle.
Dataset already unzipped at
`~/apherisfold_inputs/apherisfold_inputs/{train,val}` (10 + 17 complexes).
OF3 cloned at `~/of3/openfold-3` (pixi-managed) — leave it alone.
ApherisFold UI live on `:8080`. uv 0.11.17 available;
`~/biohackathon-work/.venv` build in progress with rdkit, transformers,
torch-CPU, gemmi, biotite, umap-learn, hdbscan. RCSB confirms **359 PDE10A
entries (Q9Y233)** for augmentation — all same UniProt, so MSA reuse is
effectively free.

## 2. Goal & success criteria

- Fine-tune OF3 on rebalanced split (10 original + ~10 moved from val +
  ~20-30 RCSB Q9Y233 augmentations); evaluate on 5–7 held-out chosen to
  span ligand chemotype clusters.
- **Win condition:** PL LDDT delta (FT − base) on held-out > 0, with IP
  LDDT delta within noise, and the gain holds on out-of-cluster held-out
  points (i.e. across-chemotype generalisation, not memorisation).
- Stretch: headline plot of PL LDDT delta vs joint similarity to train.

## 3. Decision log (newest first)

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-30 (pm) | Strategy = domain-adaptation lane, hyperparam sweeps killed | 24h hard budget, full FT is 9-12h on A100 → max 1-2 runs; data > hyperparams under that budget. |
| 2026-05-30 (pm) | Use RCSB Q9Y233 359-entry pool as primary augmentation source | Same UniProt = MSA reuse free. Mix of fragment + drug-like ligands gives chemotype breadth our 5S* fragments lack. |
| 2026-05-30 (pm) | Rebalance train/eval: ~5–7 held-out spanning clusters, rest to train | Organizer-approved. Smaller eval also speeds per-checkpoint validation. Held-out must span clusters to keep generalisation honest. |
| 2026-05-30 (pm) | My slice = similarity only; teammates own failure modes | Clean split; merge artefacts when failure-mode CSV lands. |
| 2026-05-30 (am) | Day-1 no-regression proxy = IP LDDT on held-out (no external set) | External set costs MSA-gen pass per complex; not worth day-1. |

## 4. Experiment log (newest first)

| Run ID | Date | LR | warmup | ema_decay | grad steps | crop_size | seed | PL LDDT | RMSD | Notes |
|--------|------|----|--------|-----------|-----------|-----------|------|---------|------|-------|
| baseline | _pending_ | – | – | – | 0 | – | – | | | OF3 4.0.0 base inference, 17-eval or rebalanced 5–7 |
| ft-v2 | _pending_ | 0.0003 | 50 | 0.99 | ~150 | 384 | 42 | | | Single FT on rebalanced split + augmentation, max_time ~36000s (10h) |

## 5. Environment facts

- VM IP: **31.22.104.132** (kept out of repo)
- SSH user: lyceum
- Key file: `~/.ssh/team-4`
- VM hostname: `hyd-production-vm-d8c5cmv07u62lcm1japg`, Ubuntu 22.04
- **GPU: NVIDIA A100-SXM4-80GB** (docs say H100 — wrong)
- Hub UI: http://localhost:8081 (via SSH tunnel to :8080)
- Apheris config: `/opt/apheris/config.yaml`, deploy script at
  `/opt/apheris/deploy_apherisfold`
- Dataset (already unzipped):
  `~/apherisfold_inputs/apherisfold_inputs/{train,val}` (27 cif + 28 a3m,
  including `5shr_alt1_msa.a3m`)
- OF3 source: `~/of3/openfold-3` (pixi env `openfold3-cuda12` installed)
- Similarity working dir: `~/biohackathon-work/` (venv + cache + outputs)
- uv 0.11.17 at `~/.local/bin/uv`
- pixi 0.69 at `~/.pixi/bin/pixi`
- Default OF3 fine-tune settings JSON (still authoritative):
  `{"batch_size":32,"crop_size":384,"data_seed":42,"ema_decay":0.99,
  "learning_rate":0.0003,"maximum_training_time":86400,
  "metric_to_monitor":"lddt_inter_protein_ligand","num_gradient_steps_per_epoch":16,
  "precision":"bf16","save_top_k":-1,"training_seed":42,"warmup_steps":50}`

## 6. Open threads / next steps (v2 plan execution)

- [in progress] Build VM analysis venv (uv + rdkit/transformers/gemmi etc.)
- [ ] `src/data/sequence_extract.py` — CIF → seq + ligand SMILES + pocket mask
- [ ] `src/data/embed_proteins.py` — ESM-2 (whole + pocket-only) → sim matrix
- [ ] `src/data/embed_ligands.py` — RDKit Morgan + Tanimoto + Butina
- [ ] `src/data/joint_similarity.py` — joint scoring + UMAP plot
- [ ] Run on the 27 → `per_complex_similarity.csv` + similarity_map
- [ ] `src/data/retrieve_rcsb.py` — fetch 359 Q9Y233 entries (CIFs + SMILES)
- [ ] Filter + chemotype-balance → `augmentation_candidates.csv` (~20-30)
- [ ] `src/data/msa_reuse.py` — assemble `augmentation_set/` with copied a3ms
- [ ] `src/data/splits.py` → `split_proposal.json`
- [ ] Brief team in memory.md + push to GitHub

## 7. Ideas / hypotheses parking lot

- If 2A doesn't give enough chemotype coverage: dock the 27 known PDE10A
  ligands into other PDE10A apo structures (Vina/GNINA) to synthesise
  more training poses.
- Leave-one-ligand-cluster-out evaluation as the strictest generalisation
  probe.
- Headline plot for the deck: PL LDDT delta vs joint similarity to train.
- If we have a second FT run available: targeted augmentation prioritising
  clusters where base OF3 fails (driven by failure-mode team's CSV).

## 8. Known issues / gotchas

- Apheris docs say H100 — VM has A100. Reference run will be ~9-12 h not 20h.
- `5shr` is in both the published val set AND the 359 RCSB Q9Y233 hits —
  dedupe carefully before adding it as augmentation.
- Most of the 359 RCSB entries are pre-OF3-cutoff (pretraining-seen) — the
  experiment is still valid (focused FT sharpens beyond pretraining
  exposure), but call this out in the deck honestly.
- ApherisFold Settings pane is strict JSON — no comments.
- `num_gradient_steps_per_epoch` tradeoff: lower = denser checkpoints but
  slower per-step (more validation passes).
- ICP LDDT on monomeric PDE10A is likely 0/NaN.

## 9. Presentation notes (build as we go)

- What we set out to do & why: domain adaptation on PDE10A — extend the
  10 provided fragments with similarity-curated PDE10A structures and a
  distribution-matched split.
- What we changed: dataset (augmentation + rebalanced split), not
  hyperparams.
- Evidence: per-complex PL LDDT delta + similarity score, headline scatter.
- What we'd try next: targeted augmentation driven by failure-mode CSV,
  synthetic poses for under-covered chemotype clusters.

---

## 10. SIMILARITY-LANE DELIVERABLE (team brief) — 2026-05-30 pm

**Recommendation for the single FT run we get:** use the `rebalanced`
split from `results/similarity/split_proposal.json`. That's **44 train
complexes** (10 original train + 9 reassigned from val + 25 augmentation
picks from RCSB Q9Y233) and **8 held-out eval** (one per ligand cluster,
plus 1 extra furthest-from-train). Eval shrinkage was organizer-approved
and speeds per-checkpoint validation.

**Held-out eval IDs:** `5sh0, 5sh8, 5shk, 5shr, 5sju, 5ske, 5skr, 5sku`.
These span 7 distinct ligand clusters; uncovered clusters [1,2,5,6,7,8]
are credited as covered by the augmentation picks (`covers_val_cluster_*`).

**Augmentation set:** 25 CIF+a3m pairs on the VM at
`~/biohackathon-work/similarity/augmentation_set/` (104 MB). Every a3m
is a copy of the canonical PDE10A catalytic-domain MSA (md5
`8569ccaf7d112e0d2699b3c04a1682f9`) — Apheris uses one MSA for every
single one of the 27 provided entries (verified: 27/28 a3ms are
bit-identical, the 28th is the `5shr_alt1` variant). So MSA reuse is
trivial copying, no generation needed.

**Hub UI upload procedure (whoever runs FT):**
1. Open Fine-tune → New Job → pick `openfold3 (4.0.0)`.
2. Drop the 44 train CIF+a3m pairs into Training Files. Source paths:
   - Original train (10): `~/apherisfold_inputs/apherisfold_inputs/train/`
   - Reassigned val (9): `~/apherisfold_inputs/apherisfold_inputs/val/`
     — IDs: see `split_proposal.json` → `rebalanced.reassigned_val_to_train`
   - Augmentation (25): `~/biohackathon-work/similarity/augmentation_set/`
3. Drop the 8 held-out CIF+a3m pairs into Validation Files. Source:
   `~/apherisfold_inputs/apherisfold_inputs/val/` for the IDs in
   `split_proposal.json` → `rebalanced.eval_held_out`.
4. Settings JSON: use `configs/reference.json` but bump
   `maximum_training_time` down to **36000** s (10 h) to fit the 24 h
   day budget on the A100.
5. Hit Start fine-tuning.

**Findings worth surfacing in the deck:**

- **Published 10/17 split is chemotype-stratified, almost adversarial.**
  Train and val cover *disjoint* ligand clusters (train: {0,3,4,9-14};
  val: {0,1,2,5-8} where 6 of the 7 val clusters are uncovered by train
  at Butina-cutoff 0.65). Random splits really wouldn't tell us
  anything about generalisation — the published split already enforces
  chemotype separation.
- **Pairwise ligand Tanimoto across the 27 has p50=0.17, p95=0.48** —
  classic fragment-screen heterogeneity. Don't expect strong chemotype
  transfer from one fragment to another.
- **Even with 271 (post-filter) PDE10A entries on PDB, the closest
  chemotype match to any of our val ligands has Tanimoto 0.16.** The
  augmentation set we built spans chemotype space but is not "close"
  to our val ligands; the test is whether broader PDE10A pocket
  exposure during FT helps anyway.
- **Apheris uses ONE canonical PDE10A MSA across all 27 provided
  entries** — that's why MSA reuse for ~270 PDE10A augmentation
  candidates is effectively free.
- **~88 of the 359 RCSB hits were dropped** as MSA-incompatible (different
  protein construct boundaries). Most of those are pre-2014 entries from
  early PDE10A drug-discovery campaigns. We may want to revisit those if
  the first FT run flags a need for drug-like augmentation.

**Files (mirrored to `results/similarity/` in the repo):**
- `meta_27.csv` — per-complex metadata for our 27
- `per_complex_similarity.csv` — schema-matched table for teammates
- `similarity_map.png` — UMAP of pocket ESM embeddings
- `augmentation_candidates.csv` — all 271 filtered candidates with scores
- `augmentation_picks.csv` — the 25 chosen
- `augmentation_set_manifest.csv` — what's in `augmentation_set/`
- `split_proposal.json` — original + rebalanced + 7 leave-one-cluster-out

**Coordination handoff:** once the failure-mode team produces
`results/baseline/per_complex_errors.csv` (whenever first inference
lands), merge in `notebooks/30_similarity_vs_errors.ipynb` to get the
headline plot: PL LDDT delta vs joint similarity to train, per
held-out point.
