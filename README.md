# biohackathon — OpenFold3 × PDE10A

Apheris AI-for-Co-folding hackathon. We fine-tune the public **OpenFold3
(4.0.0)** model on a small set of **PDE10A** protein–ligand co-crystals via
**ApherisFold** and ask: can ~10 liganded structures meaningfully improve OF3
pose accuracy on held-out PDE10A complexes without degrading general
performance? See [`CLAUDE.md`](./CLAUDE.md) for the project's why/what/how
and [`memory.md`](./memory.md) for current state — both files are kept up to
date and are read first every session.

## Quick start

1. Get the VM IP and the `team-4` SSH key from the team lead (BitWarden
   Send link) and place the key at `~/.ssh/team-4`.
2. Lock down the key:
   - **macOS/Linux:** `chmod 600 ~/.ssh/team-4`
   - **Windows:** `icacls "$env:USERPROFILE\.ssh\team-4" /inheritance:r /grant:r "${env:USERNAME}:R"`
3. Export the VM IP in your shell (never commit it):
   ```sh
   export VM_IP=<the ip you were given>
   ```
4. Open the Hub UI tunnel in one terminal:
   ```sh
   bash scripts/tunnel.sh
   ```
   Then load `http://localhost:8081`.
5. Get a VM shell in another terminal:
   ```sh
   bash scripts/connect.sh
   ```
6. First time only — bootstrap the dataset on the VM:
   ```sh
   bash scripts/setup_vm.sh   # run on the VM after SSH
   ```

## Running a baseline

The baseline is OF3 4.0.0 inference on the 17 held-out PDE10A complexes,
5 samples per complex, via the **Predict** page in the Hub UI. Capture
PL LDDT, IP LDDT, ICP LDDT, and pose RMSD per complex.

Details and the exact UI sequence are in [`scripts/README.md`](./scripts/README.md).

## Running a fine-tune

1. Open the Hub UI → **Fine-tune** → **+ New Job**.
2. Pick `openfold3 (4.0.0)` and give the job a name (e.g. `d0a-short`).
3. Drag-drop the 10 training CIFs (and their a3m MSAs) into Training Files.
4. Drag-drop the 17 eval CIFs into Validation Files.
5. Open the **Settings** tab and paste the contents of the matching JSON
   from `configs/` (e.g. `configs/d0a_short.json`). Per-config rationale
   and the `num_gradient_steps_per_epoch` tradeoff are in
   [`configs/README.md`](./configs/README.md).
6. Hit **Start fine-tuning**.

## Deploying fine-tuned weights

```sh
bash scripts/deploy_weights.sh <path/to/checkpoint.pt> <version-string>
# example: bash scripts/deploy_weights.sh ./of3_d0a.pt 3.0.0-d0a-short
```

This `scp`s the checkpoint to the VM, drops it under
`weights_mount/fine-tuned/`, appends an entry to `additional_weights.json`,
and runs `./deploy_apherisfold && ./deploy_apherisfold diagnose`. Reload the
UI; the new version shows in the Predict page dropdown next to the base
weights. Reminder: base UI shows `openfold3 (4.0.0)`; fine-tuned weights
register under a **separate** version-string namespace — they're not on the
same scale.

## Repo layout

```
biohackathon/
├── README.md                  this file
├── CLAUDE.md                  stable project memory (auto-loaded by Claude Code)
├── memory.md                  living state — read first each session
├── .gitignore                 excludes data, weights, keys, secrets
├── background_info/           organizer PDF, blog link, decks
├── configs/                   one JSON per recipe + README explaining each
├── src/                       data/, eval/, infer/, utils/ — Python helpers
├── scripts/                   shell scripts for VM setup, tunnel, deploy
├── eval/                      held-out + general-set PDB-ID lists
├── results/                   per-run logs, metrics, summary.csv
└── notebooks/                 exploratory analysis
```

## Results format

Every fine-tune lives under `results/runs/<run_id>/`:
- `config.json` — the exact Settings payload
- `metrics.csv` — one row per emitted checkpoint
  (`step, pl_lddt, ip_lddt, icp_lddt, mean_rmsd, ...`)
- `notes.md` — what we changed and what we saw, written as we go

`results/summary.csv` is the cross-run rollup: one row per
`(run_id, checkpoint_step, eval_set)` tuple. Build it from `metrics.csv` at
the end of each run.

## Contributing

- Small commits, PR for anything non-trivial.
- Don't commit data, weights, or anything that came over the SSH key.
- `memory.md` is the canonical place for "what's running, what's next" —
  update it after every meaningful result. Flush open threads before
  context gets compacted.
- `CLAUDE.md` should stay short and stable. If something there is wrong,
  fix it in a separate commit and call it out.
