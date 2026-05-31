# Fine-tuning configs

One JSON per recipe. Each file is exactly the payload pasted into the
ApherisFold UI's **Settings** pane when creating a fine-tuning job. The UI
parses it as strict JSON, so no inline comments — per-config rationale lives
here.

## The config we actually used: `final_fine_tuned.json`

This is the **production config** for the FT job whose checkpoint is
deployed on the team's Apheris cluster as weight version
`4.0.0-team4-aug-5sonly-v1` (the recovered `34-280.ckpt`) and whose
held-out inference numbers live in
`results/runs/prediction_fine_tuned.csv`.

```json
{
    "batch_size": 4,
    "crop_size": 384,
    "data_seed": 42,
    "ema_decay": 0.99,
    "learning_rate": 0.0003,
    "maximum_training_time": 36000,
    "metric_to_monitor": "lddt_inter_protein_ligand",
    "num_gradient_steps_per_epoch": 8,
    "precision": "bf16",
    "save_top_k": -1,
    "training_seed": 42,
    "warmup_steps": 50
}
```

### Why these specific values

| Key | Value | Why |
|---|---|---|
| `batch_size` | **4** | Empirically the largest stable batch size on the A100-80GB at `crop_size: 384` for this dataset; doubles effective throughput vs the published `batch_size: 1` reference and was what the production job ran with. |
| `crop_size` | 384 | Apheris default; protein crop fed to OF3 per training sample. |
| `learning_rate` | 3e-4 | Published Apheris PDE10A FT recipe. |
| `warmup_steps` | 50 | Published recipe. |
| `ema_decay` | 0.99 | Published recipe — exponential-moving-average anchoring to base weights to avoid catastrophic forgetting. |
| `num_gradient_steps_per_epoch` | **8** | ApherisFold writes metrics + a checkpoint every `N` grad steps. 8 gives a dense PL-LDDT curve (a fresh checkpoint every ~5 minutes on the A100) without grinding to a halt on validation. Combined with `save_top_k: -1` this produced 35 checkpoints over ~3 hours of training. |
| `save_top_k` | -1 | Keep every checkpoint — needed for the per-checkpoint PL LDDT / IP LDDT trajectory plot, and the reason we could recover the last checkpoint (`34-280.ckpt`) after the disk crash. |
| `maximum_training_time` | 36000 (10 h) | Hard hackathon budget; the training actually died at ~3 h (`[Errno 28] No space left on device` on the VM), well within the cap. |
| `metric_to_monitor` | `lddt_inter_protein_ligand` | = PL LDDT = LDDT-PLI; the primary pose-quality metric. |
| `precision` | `bf16` | Required by the published recipe; matches the OF3 4.0.0 inference path. |
| `data_seed` / `training_seed` | 42 / 42 | Deterministic shuffle + init; both seeds explicit so a re-run is reproducible. |

To reproduce the headline numbers in `results/runs/prediction_fine_tuned.csv`:
upload `dataset/{train,eval}/`, paste **this JSON** into the Hub UI's
Settings tab, hit Start.

## `reference.json` (historical anchor)

`reference.json` carries the published Apheris PDE10A recipe with
`batch_size: 1` — the recipe we used as a starting point before
empirically bumping to `batch_size: 4`. Kept in the repo as the "what
the Apheris docs say" reference.

## Other configs (exploratory, not used for the headline run)

| File | LR | EMA | Max time | Notes |
|---|---|---|---|---|
| `d0a_short.json` | 3e-4 | 0.99 | 21600 s (~6 h) | Day-1 short probe; never executed (we went straight to the augmented run). |
| `d1_ema_low.json` | 3e-4 | 0.95 | 28800 s | Hyperparam-sweep candidate; killed by the v2 plan pivot. |
| `d1_ema_high.json` | 3e-4 | 0.995 | 28800 s | Same — held in repo as design record. |
| `d1_lr_low.json` | 1e-4 | 0.99 | 28800 s | Same. |
| `d1_lr_high.json` | 6e-4 | 0.99 | 28800 s | Same. |
| `d2_augmented.json` | 3e-4 | 0.99 | 72000 s | Earlier draft of the augmented-data recipe; superseded by `final_fine_tuned.json`. |

These are kept as historical record of the v1 plan; nothing depends on them
to reproduce the headline result.

## Reproducibility

`training_seed` and `data_seed` are both 42. To probe FT variance, copy
`final_fine_tuned.json` and bump `training_seed` (e.g. 7, 1337) — leave the rest
fixed. The exact JSON above + the 35-train / 8-eval dataset reproduces
the run.
