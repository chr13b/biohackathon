# scripts/

Helper shell scripts. All require `$VM_IP` exported and the `team-4` key at
`~/.ssh/team-4`.

| Script | Where it runs | What it does |
|---|---|---|
| `connect.sh` | local | SSH into the VM as `lyceum`. |
| `tunnel.sh` | local | Forward Hub UI to `http://localhost:8081`. Hold it open. |
| `setup_vm.sh` | **on the VM** | Unzip `/apheris/apherisfold_inputs.zip`, verify all 27 CIF + a3m pairs, print GPU info. |
| `deploy_weights.sh <ckpt.pt> <version>` | local | scp a checkpoint to `weights_mount/fine-tuned/`, append it to `additional_weights.json`, run `./deploy_apherisfold` + `diagnose`. |

## End-to-end reproduction

The whole pipeline is six steps. Most run on your laptop; the FT and
inference run inside the Apheris Hub on the VM.

```
                        +---------+   +----------+   +------------+   +-------+   +-------------+   +-------+
laptop  →  (sim lane) → | filter  | → | augment  | → | rebalanced | → | Hub UI| → | deploy ckpt | → | infer |
                        | + score | → | set      | → | split.json | → | FT    | → | + inference | → | + eval|
                        +---------+   +----------+   +------------+   +-------+   +-------------+   +-------+
       src/data/                       src/data/      results/         configs/      scripts/        src/data/
       embed_*.py                      assemble_      similarity/      reference     deploy_         run_
       joint_                          augmentation_  split_           .json         weights.sh      inference_
       similarity.py                   set.py         proposal.json                                  8.py
```

### Step-by-step

1. **Open the tunnel + SSH** (two terminals):
   ```sh
   bash scripts/tunnel.sh     # terminal 1, keep open
   bash scripts/connect.sh    # terminal 2
   ```

2. **Bootstrap the VM** (once):
   ```sh
   # on the VM
   bash setup_vm.sh
   ```

3. **Run the similarity lane** (laptop or VM, both work — outputs to repo):
   ```sh
   python -m src.data.run_27               # parse 27 CIFs, embed
   python -m src.data.retrieve_rcsb        # fetch 359 RCSB Q9Y233 hits
   python -m src.data.select_augmentation  # rank + pick 16 5S* picks
   python -m src.data.assemble_augmentation_set
   python -m src.data.splits               # → results/similarity/split_proposal.json
   ```
   Outputs land in `results/similarity/`. **You can skip this step** if you
   trust the pre-committed split: it is exactly `dataset/{train,eval}/`.

4. **Upload + start the FT in the Hub UI** (one-time, ~3 h on A100):
   - Hub UI → **Fine-tune** → **+ New Job** → name it (e.g. `team4-aug-5sonly-v1`).
   - Choose weights: `openfold3 (4.0.0)`.
   - Drag-drop **all 70 files** in `dataset/train/` into **Training Files**.
   - Drag-drop **all 16 files** in `dataset/eval/` into **Validation Files**.
   - Paste **`configs/reference.json`** into the **Settings** tab.
   - Hit **Start fine-tuning**.

5. **Deploy the recovered checkpoint** (or any equivalent):
   ```sh
   bash scripts/deploy_weights.sh \
       checkpoint_of_first_run/34-280.ckpt \
       4.0.0-team4-aug-5sonly-v1
   ```
   This `scp`s the checkpoint to the VM, registers it in
   `additional_weights.json`, and runs `./deploy_apherisfold diagnose`.

6. **Run inference + eval on the 8 held-out**:
   ```sh
   # base reference (already on the Hub as openfold3 4.0.0)
   python -m src.data.run_inference_8 \
       --eval-dir dataset/eval \
       --weight-version 4.0.0 \
       --out results/inference/base_4_0_0.csv

   # fine-tuned weights we just deployed
   python -m src.data.run_inference_8 \
       --eval-dir dataset/eval \
       --weight-version 4.0.0-team4-aug-5sonly-v1 \
       --out results/inference/ft_team4_aug_5sonly_v1.csv
   ```
   The CSVs contain per-complex PL LDDT / IP LDDT / sample-wise stats.
   Compute deltas (FT − base) per held-out complex; that is the headline
   number for the deck.

## UI-driven workflows

ApherisFold's fine-tuning and inference run through the Hub UI; the docs
don't expose a CLI for either. These walkthroughs assume the tunnel is up.

### Baseline inference on the 8 eval complexes

1. Hub UI → **Predict** → **+ New Prediction**.
2. Choose weights: `openfold3 (4.0.0)`.
3. Upload the 8 eval CIFs from `dataset/eval/` (or select them from the
   previously uploaded training-job dataset).
4. Set samples-per-complex to 5 (match the published reference protocol).
5. Run; download per-complex predictions when done.
6. Compute PL LDDT, IP LDDT, ICP LDDT, and pose RMSD vs the held-out CIFs.

### Fine-tune (production config)

1. Hub UI → **Fine-tune** → **+ New Job**, name it `team4-aug-5sonly-v1`.
2. Choose weights: `openfold3 (4.0.0)`.
3. Drag-drop `dataset/train/*` into **Training Files** and `dataset/eval/*`
   into **Validation Files**. Wait for all files to turn green.
4. Paste `../configs/reference.json` into the **Settings** tab.
5. Hit **Start fine-tuning**. Track on the Fine-tune page.
6. When checkpoints emit, download via the API:
   ```sh
   curl http://localhost:8080/api/v1/fine-tune/<job_id> > results/runs/<job_id>/job.json
   ```
   then extract `metrics[*]` for the per-checkpoint PL/IP LDDT trajectory.
