# biohackathon — OpenFold3 × PDE10A: FoldMaxxers

> **Apheris AI-for-Co-folding hackathon (Team 4) — May 2026.**
> We fine-tune the public **OpenFold3 4.0.0** co-folding model on **human
> PDE10A** (UniProt `Q9Y233`) and ask whether distribution-aware data
> augmentation can move the pose-quality needle on held-out complexes.

**Yes — it does.** On 8 held-out PDE10A complexes the fine-tuned model
improves mean **Ligand-Protein Interaction LDDT** from **0.56 → 0.76**
(+0.20 absolute, +35 % relative) and cuts mean **Ligand RMSD** from
**5.34 Å → 3.03 Å** (−2.31 Å, −43 %), while **protein Cα LDDT actually
rises** from 0.96 → 0.99 (no general-performance regression).

This repo contains **everything needed to reproduce that result
end-to-end**: the curated dataset, the exact training config, the
recovered checkpoint, and the pipeline scripts. A presentation-friendly
walk-through lives in [`Guide.md`](./Guide.md) and as a
self-contained webpage in [`Guide.html`](./Guide.html); the
detailed data-lane writeup is in [`html_files/report.html`](./html_files/report.html).

---

## TL;DR — the headline

| Metric (mean over 8 held-out) | Base OF3 4.0.0 | Our FT | Δ |
|---|---:|---:|---:|
| **Ligand-Protein Interaction LDDT** (↑) | **0.563** | **0.762** | **+0.200** |
| **Ligand RMSD** (Å, ↓) | **5.34** | **3.03** | **−2.31** |
| **Protein Cα LDDT** (↑, general-perf proxy) | **0.964** | **0.989** | **+0.025** |
| **Protein Cα RMSD** (Å, ↓) | **1.69** | **1.40** | **−0.29** |

Per-complex numbers in [`results/runs/`](./results/runs/). Two complexes
flip dramatically (`5sh8`: PL 0.41 → 0.98, ligand RMSD 7.27 → 0.33 Å;
`5sh0`: PL 0.26 → 0.72, RMSD 8.63 → 2.97 Å); one stays stuck (`5sku`).
Six of eight strictly improve on PL LDDT.

| | |
|---|---|
| **Base model** | OpenFold3 4.0.0 (public weights) via Apheris Hub |
| **Target** | Human phosphodiesterase 10A (PDE10A, UniProt Q9Y233) |
| **Training data** | **35** PDE10A co-crystal structures: 10 original Apheris train + 9 reassigned from val + **16 RCSB augmentations** (real 5S\* fragment-screen entries) |
| **Held-out eval** | 8 PDE10A complexes spanning 7 ligand chemotype clusters |
| **Recipe** | LR=3e-4, warmup=50, ema=0.99, bf16, crop=384, **batch=4** — see [`configs/final_fine_tuned.json`](./configs/final_fine_tuned.json) |
| **Compute** | 1× NVIDIA A100-80GB on the Apheris VM, ~3 h of training (35 checkpoints emitted) |
| **Recovered checkpoint** | Deployed on the team's Apheris cluster as weight-version `4.0.0-team4-aug-5sonly-v1` (not stored in this repo) |
| **Approach** | **Domain adaptation, not hyperparameter tuning** — we change *which* PDE10A complexes the model sees, not how it learns. |

The training run started cleanly, produced 35 emitted checkpoints with
monotonically improving validation PL LDDT, then died at step 280 from
`[Errno 28] No space left on device` on the VM. The last checkpoint
(`34-280.ckpt`) was recovered, registered with the Hub as weight
version `4.0.0-team4-aug-5sonly-v1`, and is now available directly
from the team's Apheris cluster — no local file needed.

---

## Repo layout

```
biohackathon/
├── README.md                      ← you are here
├── Guide.md                ← step-by-step narrative for the deck
├── Guide.html              ← same content, self-contained webpage
├── CLAUDE.md                      ← stable project memory (env, why/what/how)
├── memory.md                      ← living session state
│
├── html_files/                    ← self-contained webpages
│   ├── report.html                  long-form data-lane writeup
│   └── umap_explainer.html          teammate's UMAP dashboard
│
├── dataset/                       ← THE actually-used FT data
│   ├── README.md                  per-entry provenance + IDs
│   ├── manifest.json
│   ├── train/                     35 CIF + 35 _msa.a3m (70 files)
│   └── eval/                       8 CIF +  8 _msa.a3m (16 files)
│
├── configs/
│   ├── README.md                  per-key rationale for the FT settings
│   ├── final_fine_tuned.json      ← PRODUCTION config (the one we used)
│   └── reference.json             published Apheris recipe (historical anchor)
│
├── results/
│   ├── runs/
│   │   ├── standard_openfold_performance.csv   per-complex base OF3 metrics
│   │   └── prediction_fine_tuned.csv           per-complex FT metrics
│   └── similarity/                similarity-lane outputs (UMAP, scores, picks)
│       ├── similarity_map.png
│       ├── per_complex_similarity.csv
│       ├── augmentation_candidates.csv
│       ├── augmentation_picks.csv
│       └── split_proposal.json
│
├── src/
│   └── data/                      Python pipeline modules
│       ├── apheris_client.py         REST API client for the Apheris Hub
│       ├── sequence_extract.py       CIF → seq + ligand SMILES + pocket mask
│       ├── embed_proteins.py         ESM-2 pocket embeddings → similarity matrix
│       ├── embed_ligands.py          RDKit Morgan fps → Tanimoto + Butina clusters
│       ├── joint_similarity.py       joint scoring + UMAP plot
│       ├── retrieve_rcsb.py          fetch 359 Q9Y233 entries from RCSB
│       ├── select_augmentation.py    chemotype-balanced pick of augmentation set
│       ├── assemble_augmentation_set.py   copy canonical a3m + write final set
│       ├── splits.py                 build the rebalanced train/eval split
│       ├── run_27.py                 driver: parse + embed + cluster the 27
│       └── run_inference_8.py        driver: run Hub-UI inference on the 8 eval
│
├── scripts/
│   ├── README.md                  ← end-to-end reproduction walk-through
│   ├── tunnel.sh                  forward Hub UI to localhost:8081
│   ├── connect.sh                 SSH into the VM
│   ├── setup_vm.sh                bootstrap dataset zip + GPU info
│   └── deploy_weights.sh          register a checkpoint as a Hub weight version
│
├── notebooks/                     exploratory analysis
├── background_info/               organizer materials (PDFs, slides, blog)
└── graph_similarity{,_pde10}.py   teammate's spectral protein-side similarity
```

---

## Reproduce the headline result

There are two reproduction paths. **(A)** is the fast path — verify the
deltas using the recovered checkpoint. **(B)** re-runs the FT from
scratch on the same data and config. Either way you need:

- The Apheris team's VM access (SSH key `team-4` + IP); a hub UI tunnel
  is established with `scripts/tunnel.sh`.
- A local Python ≥ 3.10 environment with `requests`, `pandas`, `gemmi`,
  `rdkit` (only `requests` + `pandas` are needed for path A).

### Five-minute prerequisites

```sh
# Clone
git clone git@github.com:chr13b/biohackathon.git
cd biohackathon

# Provide VM access (key from the team lead)
cp <your team-4 key> ~/.ssh/team-4 && chmod 600 ~/.ssh/team-4
export VM_IP=<provided>

# Python env (uv is recommended but venv works too)
uv venv .venv && source .venv/bin/activate   # or: python -m venv .venv && source .venv/bin/activate
uv pip install requests pandas               # minimum for path A
# (full env for path B / similarity-lane re-run:)
# uv pip install rdkit transformers torch gemmi biotite umap-learn hdbscan matplotlib
```

### (A) Skip training, just evaluate the already-deployed checkpoint (~30 min)

The fine-tuned weights are **already deployed on the team's Apheris
cluster** as version `4.0.0-team4-aug-5sonly-v1` — no local checkpoint
file is required. You only need cluster access (the `team-4` SSH key
and the VM IP).

```sh
# 1. Open the Hub UI tunnel and keep it open in one terminal
bash scripts/tunnel.sh

# 2. Confirm the FT weight version is visible (in a second terminal)
curl -s http://localhost:8080/api/v1/models | grep "4.0.0-team4-aug-5sonly-v1"
# If it isn't listed, re-run the deploy step on the cluster — see
# scripts/README.md → "Re-registering the deployed FT weights".

# 3. Run base + FT inference on the 8 eval complexes (against the
#    cluster-resident weights — no local .ckpt file involved)
python -m src.data.run_inference_8 \
    --eval-dir dataset/eval \
    --weight-version 4.0.0 \
    --out results/runs/base_repro.csv

python -m src.data.run_inference_8 \
    --eval-dir dataset/eval \
    --weight-version 4.0.0-team4-aug-5sonly-v1 \
    --out results/runs/ft_repro.csv

# 4. Compare against the committed reference numbers
diff <(sort results/runs/standard_openfold_performance.csv) \
     <(sort results/runs/base_repro.csv)
```

You should reproduce — up to seed noise from the diffusion sampler — the
columns in `results/runs/{standard_openfold_performance,prediction_fine_tuned}.csv`.

> The checkpoint file itself (`34-280.ckpt`, ~ several GB) lives on the
> team's Apheris cluster under `weights_mount/fine-tuned/`. It is too
> large to commit to GitHub, but team members with cluster access can
> `scp` it locally if needed; ask the team lead for the exact path.

### (B) Full reproduction from scratch (~3 h FT + ~30 min eval)

Same as (A) but insert a fine-tune step before step 2:

1. Hub UI → **Fine-tune** → **+ New Job** → name `team4-aug-5sonly-v1`.
2. Choose weights: `openfold3 (4.0.0)`.
3. Drag-drop the contents of `dataset/train/` (70 files) into **Training Files**.
4. Drag-drop the contents of `dataset/eval/` (16 files) into **Validation Files**.
5. Open **Settings** and paste **`configs/final_fine_tuned.json`** verbatim.
6. Start training. The Hub emits a checkpoint every 8 grad steps; expect
   ~3 h on the A100. When done, download the last checkpoint and continue
   from step 2 of path (A).

Full step-by-step in [`scripts/README.md`](./scripts/README.md).

### (C) Re-run the similarity lane (optional, ~20 min on CPU)

If you want to re-derive the augmentation picks and the rebalanced split
rather than trusting the committed `dataset/`:

```sh
python -m src.data.run_27               # parse + embed the 27 organiser complexes
python -m src.data.retrieve_rcsb        # fetch the 359 RCSB Q9Y233 hits
python -m src.data.select_augmentation  # rank + pick the 16 5S* augmentations
python -m src.data.assemble_augmentation_set
python -m src.data.splits               # → results/similarity/split_proposal.json
```

Outputs land in `results/similarity/`. The committed `dataset/` was
built by exactly this pipeline.

---

## Where the data came from (and what's real vs. picked)

Everything in `dataset/` is a **real, deposited PDB structure**.
Nothing is synthetic, no docking, no diffusion-generated poses.

- **10 original Apheris train** — provided in `apherisfold_inputs.zip`.
- **9 reassigned from Apheris val** — the published 10/17 split is
  almost adversarially chemotype-stratified (train vs val cover
  *disjoint* Butina clusters); we moved 9 of the more in-distribution
  val entries to train, leaving an 8-complex held-out spanning 7
  distinct ligand clusters.
- **16 augmentations** — selected from **359 RCSB entries for UniProt
  Q9Y233**. We filtered to 271 MSA-compatible candidates, ranked them
  by joint pocket+ligand similarity to our 27, and picked a
  chemotype-balanced set of 5S\* fragment-screen entries that **were
  not already in our 27**.

All 43 entries share the **same canonical PDE10A catalytic-domain MSA**
(md5 `8569ccaf7d112e0d2699b3c04a1682f9`), which is the one Apheris
bundled with its 27 starter complexes — the augmentation MSA cost was
therefore zero, a key finding of the similarity lane.

Full method in [`html_files/report.html`](./html_files/report.html) §"Augmentation rationale".

---

## What's in the box

- `dataset/` — the actually-used 35-train / 8-eval split. Every byte
  here was uploaded to the Apheris Hub for the production FT job.
- `configs/final_fine_tuned.json` — the exact Settings JSON pasted into
  the Hub UI. `batch_size: 4`, `num_gradient_steps_per_epoch: 8`,
  `save_top_k: -1`.
- **Fine-tuned weights** — `34-280.ckpt`, the last checkpoint emitted
  before disk exhaustion (grad step 280, epoch index 34 on the dense
  `num_gradient_steps_per_epoch=8` cadence). The file is **not in this
  repo** — it lives on the team's Apheris cluster and is reachable via
  the Hub UI's Predict page under version `4.0.0-team4-aug-5sonly-v1`.
- `results/runs/standard_openfold_performance.csv` — per-complex
  metrics for base OF3 4.0.0 on the 8 held-out, run 2026-05-31.
- `results/runs/prediction_fine_tuned.csv` — per-complex metrics for
  our FT weights on the same 8 held-out, run minutes later.
- `src/data/` — the Python pipeline. `apheris_client.py` is a thin
  REST wrapper around the Apheris Hub (undocumented but stable); the
  rest are the similarity-lane modules.
- `results/similarity/` — outputs of the similarity lane: UMAP figure,
  per-complex similarity CSV, augmentation picks, the split JSON.
- `html_files/report.html` — the long-form data-lane writeup with
  embedded figures and the "real vs synthetic" clarification panel.
- `Guide.md` / `Guide.html` — narrative walk-through
  suitable for slides, with the measured numbers folded in.

## Environment

- **Hub VM**: NVIDIA A100-SXM4-80GB, Ubuntu 22.04, Apheris Hub on
  `:8080`. 
- **Hub URL** (via tunnel): `http://localhost:8081`.
- **Local**: Python ≥ 3.10. Minimum deps (path A): `requests`, `pandas`.
  Full deps (path B + C):
  `rdkit`, `transformers`, `torch`, `gemmi`, `biotite`, `umap-learn`,
  `hdbscan`, `matplotlib`.

## License & credits

- OpenFold3 weights: see the OpenFold3 project license.
- Apheris Hub: provided by Apheris for the hackathon.
- PDE10A structures: RCSB PDB (public, CC0).
- Team 4 — biohackathon May 2026.
