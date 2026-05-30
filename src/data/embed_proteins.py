"""ESM-2 embeddings for protein sequences.

Two embeddings per complex:
- Whole protein (mean-pooled, all polymer residues of all chains).
- Pocket only (mean-pooled, binding-site residues from `sequence_extract`).

For 27 + ~300 PDE10A sequences, esm2_t12_35M is enough: it captures local
conservation while staying small and fast on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = "facebook/esm2_t12_35M_UR50D"


@dataclass
class ProteinEmbeddings:
    """Per-complex protein embeddings."""
    pdb_ids: list[str]
    whole: np.ndarray   # (N, d) mean-pooled over all polymer residues across chains
    pocket: np.ndarray  # (N, d) mean-pooled over pocket residues only


def _flatten_protein(chains: dict[str, str]) -> str:
    """Concatenate all chain sequences. Different chains separated by nothing;
    ESM is a per-sequence model so this gives one embedding per complex.
    For monomers / homomers / heteromers this is a pragmatic flatten."""
    return "".join(chains[c] for c in sorted(chains))


def _truncate_for_esm(seq: str, max_len: int = 1024) -> str:
    """ESM-2 has a soft token limit. Truncate from the centre out so we keep
    both N- and C-termini context. PDE10A is ~770 residues so usually no-op."""
    if len(seq) <= max_len:
        return seq
    keep_each = max_len // 2
    return seq[:keep_each] + seq[-keep_each:]


def embed_sequences(
    pdb_ids: list[str],
    sequences: list[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 4,
    device: str | None = None,
) -> np.ndarray:
    """Mean-pool ESM-2 token embeddings over residues -> (N, d)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    sequences = [_truncate_for_esm(s) for s in sequences]
    embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            toks = tokenizer(batch, padding=True, return_tensors="pt", add_special_tokens=True)
            toks = {k: v.to(device) for k, v in toks.items()}
            out = model(**toks).last_hidden_state  # (B, L, d)
            # Mean-pool over residue tokens (mask out padding AND special tokens [CLS]/[EOS])
            attn = toks["attention_mask"].unsqueeze(-1).float()
            # ESM-2 tokenizers add BOS at position 0 and EOS at the end; mask both.
            attn[:, 0, :] = 0
            # Set the EOS positions (last non-pad index per row) to 0.
            lens = toks["attention_mask"].sum(dim=1)  # (B,)
            for i, L in enumerate(lens):
                attn[i, L - 1, :] = 0
            pooled = (out * attn).sum(dim=1) / attn.sum(dim=1).clamp_min(1e-6)
            embeddings.append(pooled.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def embed_complexes(
    complexes: list,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 4,
) -> ProteinEmbeddings:
    """Return (whole, pocket) embeddings for a list of `Complex` records."""
    pdb_ids = [c.pdb_id for c in complexes]
    whole_seqs = [_flatten_protein(c.chains) for c in complexes]
    pocket_seqs = [
        c.pocket_sequence.replace("-", "") if c.pocket_sequence else
        # If pocket extraction failed, fall back to the whole sequence so
        # we still get an embedding (it just won't be pocket-specific).
        _flatten_protein(c.chains)
        for c in complexes
    ]

    print(f"[embed_proteins] embedding {len(pdb_ids)} whole sequences with {model_name}")
    whole = embed_sequences(pdb_ids, whole_seqs, model_name, batch_size)
    print(f"[embed_proteins] embedding {len(pdb_ids)} pocket sequences")
    pocket = embed_sequences(pdb_ids, pocket_seqs, model_name, batch_size)
    return ProteinEmbeddings(pdb_ids=pdb_ids, whole=whole, pocket=pocket)


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalize then dot — symmetric, ranges [-1, 1] (typically [0,1] for ESM)."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed = embeddings / norms
    return normed @ normed.T


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from src.data.sequence_extract import load_directory

    parser = argparse.ArgumentParser()
    parser.add_argument("cif_dirs", nargs="+", help="One or more directories of CIFs.")
    parser.add_argument("--out", required=True, help="Output .npz with whole/pocket/pdb_ids.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    complexes: list = []
    for d in args.cif_dirs:
        comps = load_directory(d)
        print(f"[embed_proteins] {d}: {len(comps)} complexes")
        complexes.extend(comps)

    emb = embed_complexes(complexes, model_name=args.model)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, pdb_ids=np.asarray(emb.pdb_ids), whole=emb.whole, pocket=emb.pocket)
    print(f"[embed_proteins] wrote {args.out}: whole={emb.whole.shape} pocket={emb.pocket.shape}")
