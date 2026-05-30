"""CIF -> protein sequence + ligand SMILES + pocket residue mask.

Used by every downstream similarity step. Input is a path to a .cif file
(an ApherisFold input or an RCSB download); output is a `Complex` record
with everything subsequent modules need.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import gemmi
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# Standard amino-acid three-letter set; any HETATM whose residue name is
# outside this AND outside common solvents/ions is treated as a ligand.
_AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}
_SOLVENT_IONS = {
    "HOH", "WAT", "DOD", "H2O",
    "NA", "K", "MG", "CA", "ZN", "FE", "MN", "CL", "BR", "F",
    "SO4", "PO4", "EDO", "GOL", "PEG", "PG4", "MPD", "DMS", "ACT", "FMT",
    "TRS", "EPE", "MES", "BTB", "BIS",
    "IOD", "CD", "NI", "CO", "CU", "HG", "PB", "BA", "CS",
}
_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "U", "PYL": "O",
}

POCKET_RADIUS_A = 5.0  # binding-site cutoff in Angstroms (heavy-atom distance)


@dataclass
class Ligand:
    """A non-polymer ligand extracted from a CIF."""
    ccd_id: str             # 3-letter Chemical Component Dictionary code
    chain: str
    seqid: int
    smiles: str | None      # RDKit-canonicalized SMILES (None if extraction failed)
    n_heavy_atoms: int


@dataclass
class Complex:
    """One CIF parsed into protein chains, ligand(s), and pocket residue mask."""
    pdb_id: str
    cif_path: Path
    chains: dict[str, str]                  # chain_id -> aa sequence (one-letter)
    seqres_md5: str                         # md5 of concatenated chain sequences
    ligands: list[Ligand]
    primary_ligand: Ligand | None
    pocket_residues: dict[str, list[int]]   # chain_id -> sorted residue numbers
    pocket_sequence: str                    # concatenated pocket residues, '-' between chains
    n_chains: int = field(init=False)

    def __post_init__(self) -> None:
        self.n_chains = len(self.chains)


def _is_ligand_residue(res: gemmi.Residue) -> bool:
    """Heuristic: residue is a ligand if its CCD code is not an AA and not a solvent/ion."""
    name = res.name.strip().upper()
    if name in _AA3 or name in _SOLVENT_IONS:
        return False
    # Single-atom residues are almost always ions.
    if len(res) <= 1:
        return False
    return True


def _residue_to_aa(name: str) -> str | None:
    return _3TO1.get(name.strip().upper())


def _extract_chain_sequences(model: gemmi.Model) -> dict[str, str]:
    """Per-chain one-letter protein sequence, polymer residues only, in order."""
    out: dict[str, str] = {}
    for chain in model:
        seq_chars: list[str] = []
        for res in chain:
            aa = _residue_to_aa(res.name)
            if aa is not None:
                seq_chars.append(aa)
        if seq_chars:
            out[chain.name] = "".join(seq_chars)
    return out


def _extract_ligands(model: gemmi.Model) -> list[Ligand]:
    out: list[Ligand] = []
    for chain in model:
        for res in chain:
            if not _is_ligand_residue(res):
                continue
            n_heavy = sum(1 for atom in res if atom.element.atomic_number > 1)
            if n_heavy < 4:  # too small to be a meaningful ligand
                continue
            smiles = _residue_to_smiles(res)
            out.append(
                Ligand(
                    ccd_id=res.name.strip().upper(),
                    chain=chain.name,
                    seqid=res.seqid.num,
                    smiles=smiles,
                    n_heavy_atoms=n_heavy,
                )
            )
    return out


def _residue_to_smiles(res: gemmi.Residue) -> str | None:
    """Build a small RDKit molecule from a CIF residue's atoms + bonds.

    Strategy: use RDKit's RWMol; add atoms with their element symbol; do
    not attempt to perceive bonds from coordinates here (RDKit would need
    explicit bonds anyway). Fall back to atom-only formula encoding if
    bond perception fails.
    """
    try:
        mol = Chem.RWMol()
        idx_by_atom: dict[int, int] = {}
        for i, atom in enumerate(res):
            elem = atom.element.name
            a = Chem.Atom(elem)
            idx_by_atom[i] = mol.AddAtom(a)
        # Bond perception from distance: any pair of heavy atoms within
        # element-pair covalent-radius sum is a single bond. Cheap and
        # adequate for clustering — we are not interested in stereochemistry.
        atoms = [a for a in res]
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                pi, pj = atoms[i].pos, atoms[j].pos
                d = pi.dist(pj)
                ri = atoms[i].element.covalent_r or 0.77
                rj = atoms[j].element.covalent_r or 0.77
                if d < (ri + rj) * 1.25 and d > 0.4:
                    try:
                        mol.AddBond(idx_by_atom[i], idx_by_atom[j], Chem.BondType.SINGLE)
                    except Exception:
                        pass
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            # Sanitize failure is fine for similarity purposes; keep raw.
            pass
        smi = Chem.MolToSmiles(mol, canonical=True)
        return smi if smi else None
    except Exception:
        return None


def _pocket_residue_mask(
    model: gemmi.Model,
    ligand: Ligand,
    radius: float = POCKET_RADIUS_A,
) -> dict[str, list[int]]:
    """For a chosen ligand, return per-chain residue numbers within `radius`
    Angstrom of any heavy atom of the ligand. Polymer residues only."""
    # Collect heavy-atom positions of the chosen ligand
    target_positions: list[gemmi.Position] = []
    for chain in model:
        if chain.name != ligand.chain:
            continue
        for res in chain:
            if res.seqid.num == ligand.seqid and res.name.strip().upper() == ligand.ccd_id:
                for atom in res:
                    if atom.element.atomic_number > 1:
                        target_positions.append(atom.pos)
                break
    if not target_positions:
        return {}

    r2 = radius * radius
    out: dict[str, list[int]] = {}
    for chain in model:
        keep: list[int] = []
        for res in chain:
            aa = _residue_to_aa(res.name)
            if aa is None:
                continue
            close = False
            for atom in res:
                if atom.element.atomic_number <= 1:
                    continue
                p = atom.pos
                for tp in target_positions:
                    dx = p.x - tp.x
                    dy = p.y - tp.y
                    dz = p.z - tp.z
                    if dx * dx + dy * dy + dz * dz < r2:
                        close = True
                        break
                if close:
                    break
            if close:
                keep.append(res.seqid.num)
        if keep:
            out[chain.name] = sorted(set(keep))
    return out


def _pocket_sequence_string(
    chains: dict[str, str],
    model: gemmi.Model,
    pocket: dict[str, list[int]],
) -> str:
    """Build a flat pocket sequence by collecting AA at the pocket residues.
    Chains separated by '-'. Resolves seqid -> sequence index by walking
    polymer residues in order.
    """
    parts: list[str] = []
    for chain in model:
        if chain.name not in pocket:
            continue
        wanted = set(pocket[chain.name])
        residues: list[str] = []
        for res in chain:
            aa = _residue_to_aa(res.name)
            if aa is None:
                continue
            if res.seqid.num in wanted:
                residues.append(aa)
        if residues:
            parts.append("".join(residues))
    return "-".join(parts)


def _pdb_id_from_path(p: Path) -> str:
    return p.stem.split("_")[0].lower()


def load_complex(cif_path: str | Path) -> Complex:
    """Parse a single CIF into a `Complex` record."""
    cif_path = Path(cif_path)
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    model = st[0]

    chains = _extract_chain_sequences(model)
    seqres = "".join(chains[c] for c in sorted(chains))
    seqres_md5 = hashlib.md5(seqres.encode()).hexdigest()

    ligands = _extract_ligands(model)
    # Primary ligand: the largest by heavy-atom count (drug-like > fragment > ion).
    primary = max(ligands, key=lambda L: L.n_heavy_atoms, default=None)

    pocket: dict[str, list[int]] = {}
    pocket_seq = ""
    if primary is not None:
        pocket = _pocket_residue_mask(model, primary)
        pocket_seq = _pocket_sequence_string(chains, model, pocket)

    return Complex(
        pdb_id=_pdb_id_from_path(cif_path),
        cif_path=cif_path,
        chains=chains,
        seqres_md5=seqres_md5,
        ligands=ligands,
        primary_ligand=primary,
        pocket_residues=pocket,
        pocket_sequence=pocket_seq,
    )


def load_directory(cif_dir: str | Path) -> list[Complex]:
    """Load every .cif under `cif_dir` (non-recursive) and skip dotfiles."""
    cif_dir = Path(cif_dir)
    out: list[Complex] = []
    for p in sorted(cif_dir.glob("*.cif")):
        if p.name.startswith("."):
            continue
        try:
            out.append(load_complex(p))
        except Exception as exc:
            print(f"[warn] skipped {p.name}: {exc}")
    return out


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    comps = load_directory(target)
    print(f"loaded {len(comps)} complexes from {target}")
    for c in comps[:5]:
        lig = c.primary_ligand
        print(
            f"  {c.pdb_id}: chains={list(c.chains)} seqres_md5={c.seqres_md5[:8]} "
            f"primary={lig.ccd_id if lig else None} "
            f"pocket_residues={sum(len(v) for v in c.pocket_residues.values())}"
        )
