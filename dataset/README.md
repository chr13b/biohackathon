# dataset/ — the curated FT dataset (rebalanced split)

Built from `src/data/splits.py` using `split_proposal.json` →
`rebalanced`. This is the exact data the Apheris fine-tune job
`finetune_5sncoprG6` (team4-augmented-v1) consumed.

## Contents

- `train/` — 44 train complexes (88 files: 44 `.cif` + 44 `_msa.a3m`):
  - 10 original Apheris train entries
  - 9 reassigned from the original val set (the "easier" / more
    in-distribution val complexes)
  - 25 augmentation picks from RCSB (real X-ray crystal structures of
    PDE10A from UniProt Q9Y233)
- `eval/` — 8 held-out complexes (16 files): one per discovered ligand
  cluster + 1 extra-far. All from the original Apheris val set.
- `manifest.json` — per-file provenance: source directory, PDB ID, sizes.

## Properties

- Every `_msa.a3m` is a verified copy of the canonical Apheris PDE10A
  catalytic-domain MSA (md5 `8569ccaf...`). They are bit-identical;
  Apheris uses one MSA for all 27 of their provided entries and we reuse
  it for every augmentation.
- Every `.cif` is a real, deposited Protein Data Bank structure. The
  augmentation picks were downloaded from
  `https://files.rcsb.org/download/<id>.cif` and were filtered for:
  (a) bound non-solvent ligand, (b) MSA-compatible sequence, (c) not
  already in our 27.

## Eval-set IDs

`5sh0, 5sh8, 5shk, 5shr, 5sju, 5ske, 5skr, 5sku`

These span 7 distinct ligand chemotype clusters (Butina cutoff 0.65
over Morgan-r2-2048 Tanimoto). The held-out set is designed to test
across-chemotype generalisation, not in-distribution memorisation.

## Train-set IDs (alphabetical)

`4llj, 4llk, 4lm0, 4lm1, 4lm3, 4lm4, 4mrz, 4msa, 5c28, 5sdy, 5se0,
5se5, 5se8, 5sea, 5see, 5ses, 5sf4, 5sfg, 5sfh, 5sfi, 5sfl, 5sfv,
5sg3, 5sg5, 5sgk, 5sgl, 5sh5, 5sh7, 5shu, 5si5, 5si6, 5si7, 5si8,
5sig, 5sih, 5sik, 5siq, 5siy, 5siz, 5sj0, 5sjl, 5sk4, 5skl, 5sko`

Note: `5sj0` (and the 4-series entries `4llj` through `4msa`) are
augmentation picks; the rest are either original Apheris train (10) or
reassigned-from-val (9). See `manifest.json` for the per-entry source.
