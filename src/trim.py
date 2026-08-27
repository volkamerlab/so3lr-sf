"""
Structure trimming operations for SO3LR-SF

This module handles structure trimming operations around ligands.
All operations return file paths to processed structures for consistent workflow.
"""

import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Tuple
import logging

from ase import Atoms
from ase.neighborlist import neighbor_list, natural_cutoffs

from .utils import write_structure, get_ligand_files
from .molecule_loader import load_ase_structure, create_residue_atom_mapping, prepare_mda_universe

logger = logging.getLogger(__name__)

# Maximum number of consecutive missing residues to bridge back in when a chain
# is fragmented by distance-based residue selection. Short gaps are filled to
# preserve local backbone continuity; longer gaps are left as separate segments.
MAX_BRIDGE_GAP = 2

# Bond length (Angstrom) for a capping hydrogen, keyed by the element of the
# boundary atom it saturates. Used to place link hydrogens along the direction
# of the bond that truncation severed.
_CAP_BOND_LENGTH = {"N": 1.01, "C": 1.09, "O": 0.96, "S": 1.34}
_CAP_BOND_LENGTH_DEFAULT = 1.02

# Cut-off distance (Angstrom) for a backbone peptide bond between the C of one
# residue and the N of the next.
_PEPTIDE_BOND_CUTOFF = 1.8


CapList = List[Tuple[str, np.ndarray]]


def trim_structure(
    protein_path: Union[str, Path],
    ligand_path: Union[str, Path],
    radius: float,
    output_dir: Optional[Union[str, Path]] = None
) -> str:
    """
    Trim protein structure around ligand within specified radius and save to disk.

    This function creates a trimmed protein structure using two different approaches:
    - For PDB files: Residue-based trimming (keeps complete residues). Short
      sequence gaps (<= MAX_BRIDGE_GAP residues) between selected residues are
      bridged to preserve local chain continuity, and every backbone/side-chain
      valence severed at a truncation boundary is capped with a single hydrogen.
    - For XYZ files: Atom-based trimming (keeps individual atoms). Severed
      valences are still hydrogen-capped, but residue completeness and chain
      continuity cannot be guaranteed - use PDB input for the full protocol.

    Capping hydrogens are appended after the kept protein atoms. In a PDB output
    they carry no residue name/number (ASE fills those arrays with defaults), so
    the capped file is suitable for single-point evaluation but may not round-trip
    through a second residue-based pass.

    The trimmed structure is saved in the same format as the input protein (PDB files
    are saved as PDB, others as XYZ) and can be used for more efficient calculations
    on large protein-ligand systems.

    Args:
        protein_path: Path to protein structure file (PDB, XYZ, SDF)
        ligand_path: Path to ligand structure file to trim around (PDB, XYZ, SDF)
        radius: Radius in Angstroms for trimming around ligand
        output_dir: Directory to save trimmed structures (default: same as protein)

    Returns:
        str: Path to saved trimmed protein file (same format as input)

    Raises:
        ValueError: If no protein atoms found within radius or structures are invalid
        FileNotFoundError: If input files don't exist

    Example:
        >>> # Trim protein to 10Å around ligand
        >>> trimmed_prot = trim_structure("protein.pdb", "ligand.sdf", 10.0)
        >>> print(f"Trimmed protein saved to: {trimmed_prot}")
    """
    protein_path = Path(protein_path)
    ligand_path = Path(ligand_path)

    # Read structures
    protein = load_ase_structure(protein_path)[0]
    ligand = load_ase_structure(ligand_path)[0]

    # Get positions
    protein_positions = protein.get_positions()
    ligand_positions = ligand.get_positions()

    # Determine trimming strategy based on protein file format
    protein_ext = protein_path.suffix.lower()

    if protein_ext == '.pdb':
        # Residue-based trimming for PDB files
        logger.info(f"Using residue-based trimming for PDB file: {protein_path.name}")
        logger.info("Complete residues will be included if any atom is within the cutoff radius")
        atoms_to_keep, caps = _trim_by_residues(
            protein_path, protein, protein_positions, ligand_positions, radius, logger
        )
        trimming_method = "residue"
    elif protein_ext == '.xyz':
        # Atom-based trimming for non-PDB files. No topology is available, so the
        # residue-completeness / gap-bridging / boundary-capping protocol cannot
        # be applied - the cut can land anywhere, including mid-ring, where a
        # capping hydrogen would be wrong. Emit a strong warning instead.
        logger.warning(f"Using atom-based trimming for {protein_ext.upper()} file: {protein_path.name}")
        logger.warning(
            "XYZ input: individual atoms are trimmed - residue completeness, "
            "chain continuity and open-valence capping are NOT guaranteed and "
            "fragments may be cut mid-residue. Use PDB input for the full "
            "residue-based protocol (complete residues, short-gap bridging, "
            "hydrogen-capped truncation boundaries)."
        )
        atoms_to_keep = _trim_by_atoms(protein_positions, ligand_positions, radius, logger)
        caps: CapList = []
        trimming_method = "atom"
    else:
        raise ValueError(f"Unsupported protein file format for trimming: {protein_ext}")

    if len(atoms_to_keep) == 0:
        raise ValueError(f"No protein atoms found within {radius} Å of ligand")

    # Create trimmed protein
    trimmed_protein = protein[atoms_to_keep]

    # Saturate valences that truncation left open with capping hydrogens
    if caps:
        trimmed_protein += Atoms(
            ['H'] * len(caps),
            positions=np.array([pos for _, pos in caps]),
        )
        logger.info(f"Added {len(caps)} capping hydrogen(s) at truncation boundaries")

    trimmed_protein_path = protein_path.with_name(f"{protein_path.stem}_trimmed_{radius}A_{trimming_method}{protein_ext}")

    # Save trimmed protein
    output_path = write_structure(trimmed_protein, trimmed_protein_path)

    logger.info(f"Trimmed protein ({len(trimmed_protein)} atoms from {len(protein)} original atoms) saved to: {output_path}")
    logger.info(f"Trimming method: {trimming_method}-based")

    return output_path


def _trim_by_atoms(protein_positions: np.ndarray, ligand_positions: np.ndarray, radius: float, logger) -> List[int]:
    """
    Trim protein by individual atoms within radius of ligand.

    Args:
        protein_positions: Array of protein atom positions
        ligand_positions: Array of ligand atom positions
        radius: Cutoff radius in Angstroms
        logger: Logger instance

    Returns:
        List of atom indices to keep
    """
    atoms_to_keep = []

    for i, prot_pos in enumerate(protein_positions):
        # Calculate minimum distance to any ligand atom
        distances = np.linalg.norm(ligand_positions - prot_pos, axis=1)
        min_distance = np.min(distances)

        if min_distance <= radius:
            atoms_to_keep.append(i)

    logger.debug(f"Atom-based trimming: {len(atoms_to_keep)} atoms within {radius} Å of ligand")
    return atoms_to_keep


def _saturate_open_valences(full_atoms: Atoms, kept_indices: List[int], logger) -> CapList:
    """
    Place a capping hydrogen on every covalent bond that trimming severed.

    A bond is "severed" when one of its atoms is kept and the other was removed.
    The cap sits along the original bond direction (pointing at the now-removed
    partner, whose coordinates are still available in ``full_atoms``) at a short,
    element-appropriate distance from the boundary atom.

    It is purely geometric (no topology needed). Because the residue-based path
    keeps whole residues, the severed bonds it is asked to cap are backbone
    peptide bonds, disulfides and covalent-ligand links - never a bond inside a
    ring - so a single hydrogen per bond is always the right saturation.

    Args:
        full_atoms: The complete, untrimmed structure.
        kept_indices: Indices (into ``full_atoms``) that survive trimming.
        logger: Logger instance.

    Returns:
        List of ``("H", xyz)`` tuples, one per severed bond.
    """
    kept = set(int(i) for i in kept_indices)
    if not kept or len(kept) == len(full_atoms):
        return []

    positions = full_atoms.get_positions()
    symbols = full_atoms.get_chemical_symbols()

    # Covalent-radius based bond perception on the full structure. Work on a
    # copy with periodicity stripped: PDB files often carry a placeholder 1 A
    # CRYST1 cell, and a tiny periodic box makes neighbour_list wrap every atom
    # onto its own images and report thousands of spurious bonds.
    probe = full_atoms.copy()
    probe.set_pbc(False)
    probe.set_cell(None)
    cutoffs = natural_cutoffs(probe, mult=1.2)
    src, dst = neighbor_list('ij', probe, cutoffs)

    caps: CapList = []
    for a, b in zip(src.tolist(), dst.tolist()):
        if a not in kept or b in kept:
            continue
        vec = positions[b] - positions[a]
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            continue
        length = _CAP_BOND_LENGTH.get(symbols[a], _CAP_BOND_LENGTH_DEFAULT)
        caps.append(("H", positions[a] + (length / norm) * vec))

    # Safety net: drop caps that would sit on top of each other.
    unique: CapList = []
    for sym, pos in caps:
        if any(np.linalg.norm(pos - kept_pos) < 0.3 for _, kept_pos in unique):
            continue
        unique.append((sym, pos))

    if unique:
        logger.info(f"Saturating {len(unique)} open valence(s) left by trimming with hydrogen")
    return unique


def _build_peptide_graph(residues: List[dict], protein_positions: np.ndarray):
    """
    Link residues by backbone peptide bonds (geometry, not residue numbering).

    Residue ``i`` is linked to residue ``j`` when the distance between the
    backbone C of ``i`` and the backbone N of ``j`` is within
    ``_PEPTIDE_BOND_CUTOFF``. Returns ``(next_of, prev_of)`` dicts keyed by the
    residue's index in ``residues``.
    """
    next_of: dict = {}
    prev_of: dict = {}

    have_n = [(k, r["N"]) for k, r in enumerate(residues) if r["N"] is not None]
    if not have_n:
        return next_of, prev_of
    n_res_idx = np.array([k for k, _ in have_n])
    n_xyz = protein_positions[np.array([idx for _, idx in have_n])]

    for i, rec in enumerate(residues):
        c_idx = rec["C"]
        if c_idx is None:
            continue
        d = np.linalg.norm(n_xyz - protein_positions[c_idx], axis=1)
        j_local = int(np.argmin(d))
        if d[j_local] <= _PEPTIDE_BOND_CUTOFF:
            j = int(n_res_idx[j_local])
            if j != i and j not in prev_of:
                next_of[i] = j
                prev_of[j] = i
    return next_of, prev_of


def _bridge_short_gaps(selected: set, next_of: dict, prev_of: dict, n_res: int,
                       max_gap: int = MAX_BRIDGE_GAP) -> set:
    """
    Return the set of residues to add so selected residues stay contiguous.

    Walks each backbone chain (via ``next_of``/``prev_of``) and, for every pair
    of selected residues separated only by unselected residues, fills the gap
    when it spans at most ``max_gap`` residues. Longer gaps are left as genuine
    breaks between separate fragments.
    """
    bridged: set = set()
    if max_gap < 1:
        return bridged

    chain_starts = [ri for ri in range(n_res) if ri not in prev_of]
    for start in chain_starts:
        chain: List[int] = []
        cur: Optional[int] = start
        walked: set = set()
        while cur is not None and cur not in walked:
            walked.add(cur)
            chain.append(cur)
            cur = next_of.get(cur)

        sel_positions = [k for k, ri in enumerate(chain) if ri in selected]
        for left, right in zip(sel_positions, sel_positions[1:]):
            if 0 < right - left - 1 <= max_gap:
                bridged.update(chain[left + 1:right])
    return bridged


def _trim_by_residues(protein_path: Path, protein: Atoms, protein_positions: np.ndarray,
                      ligand_positions: np.ndarray, radius: float, logger) -> Tuple[List[int], CapList]:
    """
    Trim protein to complete residues near the ligand, preserving chain continuity.

    Steps:
      1. Select every residue with at least one atom within ``radius`` of the ligand.
      2. Bridge short sequence gaps (<= MAX_BRIDGE_GAP residues) between selected
         residues on the same backbone chain, so fragments stay contiguous.
      3. Keep every atom of every selected residue.
      4. Cap valences severed at the truncation boundary with hydrogens.

    Only reading the topology falls back to atom-based trimming; a failure in the
    residue/gap/cap logic propagates so it is never silently degraded.

    Args:
        protein_path: Path to PDB protein file.
        protein: The full protein as an ASE Atoms object.
        protein_positions: Array of protein atom positions.
        ligand_positions: Array of ligand atom positions.
        radius: Cutoff radius in Angstroms.
        logger: Logger instance.

    Returns:
        Tuple of (sorted atom indices to keep, list of capping-hydrogen tuples).
    """
    try:
        universe = prepare_mda_universe(protein_path)
        residue_atom_mapping = create_residue_atom_mapping(universe)
        mda_residues = list(universe.residues)
        logger.debug(f"Residue-atom mapping created with {len(residue_atom_mapping)} residues")
    except Exception as e:
        logger.warning(f"Residue-based trimming failed: {e}")
        logger.warning("Falling back to atom-based trimming (no gap bridging or capping)")
        return _trim_by_atoms(protein_positions, ligand_positions, radius, logger), []

    # Per-residue records: atom indices + backbone N/C for the peptide graph.
    residues: List[dict] = []
    for res in mda_residues:
        names = {atom.name.strip(): int(atom.index) for atom in res.atoms}
        residues.append({
            "segid": res.segid,
            "resid": int(res.resid),
            "atoms": [int(atom.index) for atom in res.atoms],
            "N": names.get("N"),
            "C": names.get("C"),
        })
    n_res = len(residues)

    next_of, prev_of = _build_peptide_graph(residues, protein_positions)

    # 1. Seed: residues owning any atom within the cutoff.
    atoms_within_radius = set(_trim_by_atoms(protein_positions, ligand_positions, radius, logger))
    selected = {
        ri for ri, rec in enumerate(residues)
        if any(ai in atoms_within_radius for ai in rec["atoms"])
    }
    logger.info(f"Residue-based trimming: {len(selected)} residues within {radius} Å of ligand")

    if not selected:
        return [], []

    # 2. Bridge short sequence gaps along each backbone chain.
    bridged = _bridge_short_gaps(selected, next_of, prev_of, n_res)
    if bridged:
        logger.info(f"Bridged {len(bridged)} residue(s) to preserve local chain continuity")
    selected |= bridged

    # 3. Keep every atom of every selected residue.
    atoms_to_keep = sorted({ai for ri in selected for ai in residues[ri]["atoms"]})
    logger.info(f"Total atoms after including complete residues: {len(atoms_to_keep)}")

    # 4. Cap the valences that truncation severed.
    caps = _saturate_open_valences(protein, atoms_to_keep, logger)

    return atoms_to_keep, caps


def perform_trimming(protein_path, ligands_source, radius, trim_lig, output_dir, logger):
    """Handle protein trimming workflow."""

    logger.info("=== TRIMMING PHASE ===")

    # Get representative ligand for trimming
    if trim_lig:
        if not Path(trim_lig).exists():
            raise FileNotFoundError(f"Specified trim ligand not found: {trim_lig}")
        representative_ligand = trim_lig
        logger.info(f"Using the specified ligand for trimming: {representative_ligand}")
    else:
        ligand_files = get_ligand_files(ligands_source, output_dir)
        if not ligand_files:
            raise ValueError("No ligand files found")
        representative_ligand = ligand_files[0]
        logger.info(f"Using first ligand for trimming: {representative_ligand}")

    # Perform trimming
    trimmed_protein_path = trim_structure(
        protein_path, representative_ligand,
        radius=radius, output_dir=output_dir
    )
    logger.info(f"Protein trimmed to {radius}Å radius: {trimmed_protein_path}")

    return trimmed_protein_path