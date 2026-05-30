# CLAUDE.md — Project Memory (auto-loaded every session)

> Keep this file short and stable. The living, frequently-updated state lives in
> `memory.md`. **At the start of every session, read `memory.md`.**

## WHY
Hackathon project: fine-tune the public **OpenFold3 (OF3)** co-folding model on
**PDE10A** protein–ligand structures using **ApherisFold**, and show we can
correct systematic pose errors for this target/chemotype without degrading
general performance.

The practical question we are answering:
> Can a focused fine-tune on PDE10A — drawing on the ~10 provided complexes plus
> augmentation from the ~359 RCSB Q9Y233 entries and distribution-matched splits —
> meaningfully improve OF3 pose accuracy on held-out PDE10A complexes?

**Framing:** organizers have explicitly framed this as a *domain adaptation*
problem (graph/embedding-based similarity matching, distribution-aware splits,
optional synthetic data) — not a hyperparameter tuning problem.

**My ownership (Chris):** the similarity / distribution lane — build the
distributional view of the 27 provided complexes + 359 RCSB Q9Y233 entries,
propose a rebalanced train/eval split, prepare augmentation CIFs + a3ms for
upload to the Apheris UI.

**Other teammates' ownership:** failure-mode characterisation of base OF3 on
the held-out complexes (will emerge from the first runs, not pre-known).

## WHAT
- **Target:** human PDE10A
- **Base model:** OpenFold3, public weights (late 2025), version `openfold3 (4.0.0)`
- **Data:** 27 structures from Roche PDE10A 2022 set, held out from OF3 training.
  - Train (10): 5SDY, 5SIQ, 5SI7, 5SIG, 5SI5, 5SI8, 5SIY, 5SG5, 5SGL, 5SIH
  - Eval (17, no gradient): 5SH0, 5SE0, 5SHR, 5SJL, 5SH8, 5SF4, 5SFG, 5SE5,
    5SHK, 5SEE, 5SFL, 5SJU, 5SKE, 5SKU, 5SKO, 5SEA, 5SKR
  - The `5S` PDB-code block is characteristic of **XChem / SGC crystallographic
    fragment screens** at Diamond — expect small, chemotype-clustered ligands
    at the cAMP/cGMP pocket. Implication: random splits are effectively
    similar-chemotype splits; pose RMSD on these tiny ligands is noisy.
  - **Augmentation pool:** RCSB full-text search for UniProt `Q9Y233` returns
    **359 PDB entries** (all human PDE10A), mixing the 5S\* fragments with
    older drug-discovery-era inhibitor co-crystals (2OU\*, 2Y0J, 2ZMF, 3UI7,
    3WI2, 3WS8/9, 4AEL, 4AJD, 4P0N, 4TPM, etc.). All share the same UniProt
    entry, so **MSAs reused from our existing 27 a3ms cover the augmentation
    set for ~free** — sequence-md5 check + copy the closest matching a3m.
  - Dataset archive on the VM: `/apheris/apherisfold_inputs.zip`
- **Primary metrics:** pose RMSD on held-out structures; Protein–Ligand LDDT (PL LDDT),
  Inter-Chain Protein LDDT (ICP LDDT), Intra-Protein LDDT (IP LDDT). Compare
  fine-tuned vs base (Step 0) — watch for general-performance regression.
  - **PL LDDT** = ApherisFold `lddt_inter_protein_ligand` = literature
    **LDDT-PLI** (CASP15/16 ligand-protein interface contacts, post-CASP15
    penalty for false contacts). Default `metric_to_monitor`.
  - **ICP LDDT** = inter-chain protein LDDT — likely 0/NaN on monomeric PDE10A;
    confirm on first run.
  - **IP LDDT** = intra-protein LDDT — general-performance proxy on the
    protein side.

## HOW (environment + commands)
- **Hardware:** single **NVIDIA A100-SXM4-80GB** (the Apheris docs say H100 — they're wrong; the VM has an A100). Reference run is ~9–12 h on A100 for the published protocol; budget the full hackathon day as 24 h.
- **VM access** (SSH user `lyceum`, key `~/.ssh/team-4`; export `VM_IP` in your shell):
  - Shell: `ssh -i ~/.ssh/team-4 lyceum@$VM_IP`
  - Hub UI tunnel: `ssh -i ~/.ssh/team-4 -N -L 8081:localhost:8080 lyceum@$VM_IP`
    then open `http://localhost:8081`
- **Reference fine-tuning hyperparameters (PDE10A):**
  learning_rate 0.0003 · warmup_steps 50 · ema_decay 0.99 · ~350 total grad steps ·
  templates disabled · MSAs as baseline OF3.
- **Weight version-strings (two independent namespaces, easy to confuse):**
  - **Base** OF3 appears in the UI dropdown as `openfold3 (4.0.0)`.
  - **Fine-tuned** weights register in `additional_weights.json` under a
    separate `version` field (Apheris docs use `3.0.0-fine-tuned` as the
    example). The two numbers are not on the same scale.
- **Deploy custom weights:** download checkpoint → rename to `of3_ft3_v1.pt` →
  `weights_mount/fine-tuned/of3_ft3_v1.pt` → register in `additional_weights.json` →
  `./deploy_apherisfold` then `./deploy_apherisfold diagnose` → use in Predict page.
- Background docs are in `./background_info/` on this box; dataset zip lives
  only on the VM at `/apheris/apherisfold_inputs.zip`.

## CONVENTIONS
- Every experiment gets an entry in `memory.md` (run id, hyperparams, metric, notes).
- Don't commit data, weights, or secrets (keys, BitWarden links). See `.gitignore`.
- Prefer small, reviewable commits; open a PR for anything teammates should see.
- VM working dir for the similarity lane: `~/biohackathon-work/` with
  `.venv` (uv-managed Python 3.10), `cache/` (RCSB downloads), `similarity/`
  (outputs). Outputs mirror back to repo `results/similarity/`.

## MEMORY PROTOCOL (survives compaction)
1. **Session start:** read `memory.md` for current state before acting.
2. **After any meaningful result or decision:** append it to `memory.md`.
3. **Before context compaction / when context is getting long:** flush the current
   plan, open threads, and latest results into `memory.md` so nothing is lost.
4. Treat `memory.md` as the source of truth across sessions and teammates.
