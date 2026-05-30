# Fine-tuning configs

One JSON per recipe. Each file is exactly the payload pasted into the
ApherisFold UI's **Settings** pane when creating a fine-tuning job. The UI
parses it as strict JSON, so no inline comments — per-config rationale lives
here.

## The `num_gradient_steps_per_epoch` tradeoff (read me first)

ApherisFold writes metrics + a checkpoint every `num_gradient_steps_per_epoch`
gradient steps. The default is **16**. Lower values give us a denser
PL-LDDT-vs-step curve (better for early stopping), but every "epoch" runs a
full validation pass, so smaller values **slow wall-clock per-step** and
**fill disk faster**.

What we picked and why:

- **`8`** — for all "long" runs (`reference.json`, the D1 grid,
  `d2_augmented.json`). Halves the validation cadence vs default, doubling
  the checkpoint resolution at a moderate throughput cost. Good enough to
  read off an early-stopping point without inflating run-time by much.
- **`4`** — only for `d0a_short.json`. The whole point of D0a is to learn the
  early-shape of the curve in 80–120 grad steps; we need ~20–30 checkpoints
  across that span, so we accept the validation-pass overhead.

`save_top_k: -1` keeps every checkpoint. That's necessary for the D3
generalisation-guard analysis (we re-evaluate at every checkpoint, not just
the final one). Plan around the disk consequence.

## Per-config rationale

| File | Lane | LR | EMA | Max time | Notes |
|---|---|---|---|---|---|
| `reference.json` | D0b | 3e-4 | 0.99 | 86400 s (~24 h) | Published Apheris recipe; unattended anchor run. |
| `d0a_short.json` | D0a | 3e-4 | 0.99 | 21600 s (~6 h) | Day-1 directional probe at ~80–120 grad steps. Dense checkpoints. **Gates D1/D2 decisions.** |
| `d1_ema_low.json` | D1 | 3e-4 | 0.95 | 28800 s (~8 h) | Less anchoring to base — does the FT model move further toward PDE10A at the cost of general performance? |
| `d1_ema_high.json` | D1 | 3e-4 | 0.995 | 28800 s (~8 h) | More anchoring — safer, smaller PL-LDDT gain expected. |
| `d1_lr_low.json` | D1 | 1e-4 | 0.99 | 28800 s (~8 h) | Stability check — should reach a similar PL-LDDT shape, more slowly. |
| `d1_lr_high.json` | D1 | 6e-4 | 0.99 | 28800 s (~8 h) | Faster movement — may overshoot or diverge; informative either way. |
| `d2_augmented.json` | D2 | 3e-4 | 0.99 | 72000 s (~20 h) | Same recipe as reference, but **fed the 10 PDE10A train set + 20–40 augmentation complexes (with their own a3m MSAs)**. **MSA generation for augmentation complexes must be running before this job starts.** |

## Decision gates (read off D0a)

| D0a outcome | Action |
|---|---|
| PL LDDT improving on eval **and** IP LDDT holding | Launch D1 grid sequentially; queue D0b. |
| PL LDDT flat | Skip D1, prioritise D2 (data is the bottleneck). |
| IP LDDT degrading materially | Run `d1_ema_high.json` and `d1_lr_low.json` first. |

## Reproducibility

`training_seed` and `data_seed` are both 42 across configs. To probe
fine-tuning variance later, copy any config and bump `training_seed` to a
different value (e.g. 7, 1337) — leave the rest fixed.
