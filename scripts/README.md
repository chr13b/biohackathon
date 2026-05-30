# scripts/

Helper shell scripts. All require `$VM_IP` exported and the `team-4` key at
`~/.ssh/team-4`.

| Script | Where it runs | What it does |
|---|---|---|
| `connect.sh` | local | SSH into the VM as `lyceum`. |
| `tunnel.sh` | local | Forward Hub UI to `http://localhost:8081`. Hold it open. |
| `setup_vm.sh` | **on the VM** | Unzip `/apheris/apherisfold_inputs.zip`, verify all 27 CIF + a3m pairs, print GPU info. |
| `deploy_weights.sh <ckpt.pt> <version>` | local | scp a checkpoint to `weights_mount/fine-tuned/`, append it to `additional_weights.json`, run `./deploy_apherisfold` + `diagnose`. |

## UI-driven workflows

ApherisFold's fine-tuning and inference run through the Hub UI; the docs
don't expose a CLI for either. These walkthroughs assume the tunnel is up.

### Baseline inference on the 17 eval complexes

1. Hub UI → **Predict** → **+ New Prediction**.
2. Choose weights: `openfold3 (4.0.0)`.
3. Upload the 17 eval CIFs (or select them from the previously uploaded
   training-job dataset).
4. Set samples-per-complex to 5 (match the published reference protocol).
5. Run; download per-complex predictions when done.
6. Compute PL LDDT, IP LDDT, ICP LDDT, and pose RMSD vs the held-out CIFs.
   Save results to `results/runs/baseline/` with a `config.json` and a
   `metrics.csv`.

### Fine-tune (any config)

1. Hub UI → **Fine-tune** → **+ New Job**, name it after the config
   (e.g. `d0a-short`).
2. Choose weights: `openfold3 (4.0.0)`.
3. Drag-drop the 10 training CIFs (+a3m) into **Training Files** and the 17
   eval CIFs (+a3m) into **Validation Files**. Wait for all files to turn
   green (MSAs cache).
4. Open the **Settings** tab and paste the contents of the matching JSON
   from `../configs/` (e.g. `../configs/d0a_short.json`).
5. **Start fine-tuning**. Track on the Fine-tune page.
6. When checkpoints emit, download metrics for each and record them in
   `results/runs/<run_id>/metrics.csv`. Same for the final checkpoint
   if you want to deploy it for inference (see `deploy_weights.sh`).
