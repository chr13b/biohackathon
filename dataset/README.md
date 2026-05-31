# dataset/ — the actually-used FT dataset (5S\*-only)

This is the **exact** data consumed by our successful (then disk-crashed) fine-tune
run `team4-aug-5sonly-v1`. It is the 5S\*-only subset of the originally proposed
44-train rebalanced split — the 9 non-5S\* augmentation entries (4LL\*, 4LM\*,
4MRZ, 4MSA, 5C28) were dropped after they triggered `MonitorDatasetGeneration`
failures, and were never seen by the model that produced the recovered
checkpoint `checkpoint_of_first_run/34-280.ckpt`.

## Contents

- `train/` — **35 train complexes** (70 files: 35 `.cif` + 35 `_msa.a3m`):
  - 10 original Apheris train entries
  - 9 reassigned from the original Apheris val set
  - 16 augmentation picks from RCSB (real X-ray crystal structures of human
    PDE10A from UniProt Q9Y233, all 5S\* fragment-screen entries)
- `eval/` — **8 held-out complexes** (16 files):
  one per discovered ligand cluster + 1 extra-far. All from the original
  Apheris val set, never seen during training.
- `manifest.json` — per-file provenance: source bucket, PDB ID, sizes.

## Properties

- Every `_msa.a3m` is a verified copy of the canonical Apheris PDE10A
  catalytic-domain MSA (md5 `8569ccaf7d112e0d2699b3c04a1682f9`). They are
  bit-identical; Apheris uses one MSA for all 27 of their provided entries
  and we reuse it for every augmentation — see `src/data/sequence_extract.py`
  and the rationale in `report.html`.
- Every `.cif` is a real, deposited Protein Data Bank structure (no synthetic
  poses). The augmentation picks were downloaded from
  `https://files.rcsb.org/download/<id>.cif` and were filtered for:
  (a) bound non-solvent ligand, (b) MSA-compatible catalytic-domain sequence,
  (c) not already in our 27.

## Eval-set IDs

```
5sh0  5sh8  5shk  5shr  5sju  5ske  5skr  5sku
```

These span 7 distinct ligand chemotype clusters (Butina cutoff 0.65 over
Morgan-r2-2048 Tanimoto). The held-out set is designed to test
across-chemotype generalisation, not in-distribution memorisation.

## Train-set IDs (alphabetical)

```
5sdy  5se0  5se5  5se8  5sea  5see  5ses  5sf4  5sfg  5sfh
5sfi  5sfl  5sfv  5sg3  5sg5  5sgk  5sgl  5sh5  5sh7  5shu
5si5  5si6  5si7  5si8  5sig  5sih  5sik  5siq  5siy  5siz
5sj0  5sjl  5sk4  5skl  5sko
```

Per-entry provenance (original-train / reassigned-from-val / augmentation):
see `manifest.json`.

## How to consume this dataset

For the Apheris Hub UI:
1. Drag-drop `train/*.cif` + `train/*_msa.a3m` into **Training Files**.
2. Drag-drop `eval/*.cif` + `eval/*_msa.a3m` into **Validation Files**.
3. Paste `configs/reference.json` into the **Settings** tab.
4. Hit **Start fine-tuning**.

The reproduced run will produce a checkpoint equivalent to (within seed
variability) `checkpoint_of_first_run/34-280.ckpt`.
