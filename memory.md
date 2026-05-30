# memory.md — Living Project State

> Source of truth across sessions, teammates, and context compaction.
> **Read this first each session. Update it whenever something changes — and always
> flush open threads here before context gets compacted.**

_Last updated: 2026-05-30_
_Updated by: Claude Code session 1 (Opus 4.7)_

---

## 1. Current status (one paragraph)

Plan approved with edits. Repo scaffolding being built locally on Chris's
Windows box (Python 3.12, torch 2.2.1, no local GPU). VM access is resolved
— team-4 SSH key is in place, VM IP is known to the user (export as `VM_IP`
in the shell that runs `scripts/setup_vm.sh`). Strategy is **D0a fast probe
→ gate D1/D2 off D0a → D0b in background → D3 free on every checkpoint**.
Day-1 no-regression proxy: **IP LDDT on the held-out 17 eval set** (no
external general set yet — defer until eval pipeline is solid). Nothing has
been run on the VM yet.

## 2. Goal & success criteria

- Fine-tune OF3 on 10 PDE10A complexes; evaluate on 17 held-out complexes.
- **Win condition:** lower pose RMSD / higher PL LDDT on eval set vs base
  (Step 0), with **IP LDDT on the same 17 eval set holding** (day-1 proxy
  for no regression).
- Stretch: characterise *why* it worked, and what fails. Show a per-complex
  breakdown vs ligand similarity to the train set.

## 3. Decision log (newest first)

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-05-30 | Day-1 no-regression proxy = IP LDDT on the held-out 17, not an external general set | External set needs MSA generation per complex; not worth the day-1 cost. Add later only if pipeline is solid. |
| 2026-05-30 | D1 and pivots gate off D0a (80–120 step short run), NOT off D0b (20 h reference) | Don't burn 20 h to learn the direction; D0a's short run answers "does FT move PL LDDT?" in a few hours. |
| 2026-05-30 | Adopt 5-direction strategy: D0a fast probe, D0b unattended reference, D1 EMA/LR grid, D2 similarity-based augmentation, D3 free generalisation guard | Best value/hour on 1× H100 given ~hackathon time. Each direction has a hypothesis + falsifier; see plan file. |

## 4. Experiment log (newest first)

| Run ID | Date | LR | warmup | ema_decay | grad steps | crop_size | seed | PL LDDT (val) | RMSD | Notes / outcome |
|--------|------|----|--------|-----------|-----------|-----------|------|---------------|------|-----------------|
| baseline | | – | – | – | 0 | – | – | | | OF3 4.0.0 public weights, no fine-tune (Step 0 reference) — to be run |
| d0a-short | | 0.0003 | 50 | 0.99 | 80–120 | 384 | 42 | | | Directional probe; gates D1/D2 decisions |
| d0b-ref | | 0.0003 | 50 | 0.99 | ~350 | 384 | 42 | | | Reference reproduction; unattended ~20 h |

## 5. Environment facts (fill once, then stable)

- VM IP: known to user; export as `VM_IP` in shell (not committed)
- SSH user: lyceum
- Key file: `~/.ssh/team-4` (present on this box; may need Windows ACL fix
  via `icacls` so OpenSSH accepts it)
- Hub UI: `http://localhost:8081` (via tunnel to :8080)
- Dataset: `/apheris/apherisfold_inputs.zip` on the VM (NOT on this box)
- Background docs: `./background_info/` on this box
- GPU: 1× H100 on the VM; no GPU locally
- OF3 weights version in UI (base): `openfold3 (4.0.0)`
- Custom weights register under a separate version-string namespace in
  `additional_weights.json` (Apheris example: `3.0.0-fine-tuned`)
- Default settings JSON seen in UI:
  `{"batch_size":32,"crop_size":384,"data_seed":42,"ema_decay":0.99,
  "learning_rate":0.0003,"maximum_training_time":86400,
  "metric_to_monitor":"lddt_inter_protein_ligand","num_gradient_steps_per_epoch":16,
  "precision":"bf16","save_top_k":-1,"training_seed":42,"warmup_steps":50}`
- Local env: Python 3.12.0, torch 2.2.1, numpy 1.26.2. No rdkit, biopython,
  gemmi, or CUDA locally; install only if doing offline checkpoint analysis.

## 6. Open threads / next steps (TODO)

- [ ] Fix Windows ACL on `~/.ssh/team-4` (`icacls`) so OpenSSH accepts it
- [ ] First SSH login to VM; unzip `/apheris/apherisfold_inputs.zip`;
  verify 27 CIFs + 27 a3m MSAs
- [ ] Confirm Hub UI tunnel + OF3 (4.0.0) selectable on Predict page
- [ ] Run **baseline inference** (5 samples × 17 eval) and capture
  PL/IP/ICP LDDT + pose RMSD → `results/runs/baseline/`
- [ ] Launch **D0a short fine-tune** (~80–120 steps, reference hyperparams,
  dense checkpointing) → read decision gates
- [ ] Queue **D0b reference reproduction** unattended on the H100
- [ ] If D0a green: launch **D1 grid** sequentially after D0b kicks off
- [ ] If D2 likely after D0a: start **MSA generation for ~20–40 augmentation
  complexes** as a separate background task on the VM
- [ ] Build `results/summary.csv` from every (run_id, checkpoint, eval set)
  pair as runs complete

## 7. Ideas / hypotheses parking lot

- D1 axes worth exploring inside the EMA × LR grid:
  - `ema_decay`: 0.95 (less anchoring) vs 0.995 (more anchoring)
  - `learning_rate`: 1e-4 (stability) vs 6e-4 (faster movement)
  - shorter warmup tied to lower total step count
- D2 retrieval ideas:
  - protein side: Foldseek over PDE family for similar pockets
  - ligand side: Tanimoto neighbours of the 10 training ligands in
    PDB Chem Component Dictionary
  - the rest of the XChem 5S* PDE10A fragment set outside our 27
- Inference robustness:
  - sample N=10 or 20 per complex (vs default 5) and use confidence-
    weighted ranking — does pose RMSD vs N saturate at 5?
- Visualisation candidate:
  - 5SH8 (blog's anecdote — fine-tuned model "combined 5SDY + 5SI7")

## 8. Known issues / gotchas

- Windows `chmod 600` doesn't actually restrict NTFS inheritance; OpenSSH
  on Windows may reject `team-4` until ACLs are tightened via `icacls`.
- ApherisFold Settings pane is strict JSON — no comments. Per-config
  rationale lives in `configs/README.md`, not inline.
- `num_gradient_steps_per_epoch` default is 16; lower it (~8) for short
  runs to get a denser checkpoint curve, at the cost of more validation
  passes per grad step — see `configs/README.md`.
- ICP LDDT on monomeric PDE10A is likely 0/NaN — don't read it as a signal.

## 9. Presentation notes (build as we go)

- What we set out to do & why:
- What we changed:
- Evidence (numbers, curves, a concrete prediction/visualisation):
- What we'd try next:
