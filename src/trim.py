"""
Structure trimming operations for SO3LR-SF

This module handles structure trimming operations around ligands.
All operations return file paths to processed structures for consistent workflow.
"""

import numpy as np
from pathlib import Path
from typing import Union, List, Optional
import logging

from .utils import write_structure, get_ligand_files
from .molecule_loader import load_ase_structure, create_residue_atom_mapping, prepare_mda_universe

logger = logging.getLogger(__name__)


def trim_structure(
    protein_path: Union[str, Path],
    ligand_path: Union[str, Path],
    radius: float,
    output_dir: Optional[Union[str, Path]] = None
) -> str:
    """
    Trim protein structure around ligand within specified radius and save to disk.

    This function creates a trimmed protein structure using two different approaches:
    - For PDB files: Residue-based trimming (keeps complete residues)
    - For XYZ/SDF files: Atom-based trimming (keeps individual atoms)

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
        atoms_to_keep = _trim_by_residues(protein_path, protein_positions, ligand_positions, radius, logger)
        trimming_method = "residue"
    elif protein_ext == '.xyz':
        # Atom-based trimming for non-PDB files
        logger.warning(f"Using atom-based trimming for {protein_ext.upper()} file: {protein_path.name}")
        logger.warning("Individual atoms will be trimmed - residues may be incomplete!")
        logger.warning("For complete residue trimming, use PDB format input files")
        atoms_to_keep = _trim_by_atoms(protein_positions, ligand_positions, radius, logger)
        trimming_method = "atom"
    else:
        raise ValueError(f"Unsupported protein file format for trimming: {protein_ext}")

    if len(atoms_to_keep) == 0:
        raise ValueError(f"No protein atoms found within {radius} Å of ligand")

    # Create trimmed protein
    trimmed_protein = protein[atoms_to_keep]

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


def _trim_by_residues(protein_path: Path, protein_positions: np.ndarray, ligand_positions: np.ndarray,
                     radius: float, logger) -> List[int]:
    """
    Trim protein by complete residues if any atom in the residue is within radius of ligand.

    Args:
        protein_path: Path to PDB protein file
        protein_positions: Array of protein atom positions
        ligand_positions: Array of ligand atom positions
        radius: Cutoff radius in Angstroms
        logger: Logger instance

    Returns:
        List of atom indices to keep (complete residues)
    """
    try:
        # Create MDAnalysis universe to get residue information
        universe = prepare_mda_universe(protein_path)
        residue_atom_mapping = create_residue_atom_mapping(universe)
        logger.debug(f"Residue-atom mapping created with {len(residue_atom_mapping)} residues")

        # First find individual atoms within radius
        atoms_within_radius = _trim_by_atoms(protein_positions, ligand_positions, radius, logger)

        # Then identify which residues these atoms belong to
        residues_to_keep = set()
        for residue_id, atom_indices in residue_atom_mapping.items():
            # Check if any atom from this residue is within radius
            if any(atom_idx in atoms_within_radius for atom_idx in atom_indices):
                residues_to_keep.add(residue_id)
        logger.info(f"Residue-based trimming: {len(residues_to_keep)} complete residues selected")

        # Finally, include ALL atoms from selected residues
        atoms_to_keep = []
        for residue_id in residues_to_keep:
            atoms_to_keep.extend(residue_atom_mapping[residue_id])

        # Sort atom indices to maintain order
        atoms_to_keep = sorted(set(atoms_to_keep))
        logger.info(f"Total atoms after including complete residues: {len(atoms_to_keep)}")

        return atoms_to_keep

    except Exception as e:
        logger.warning(f"Residue-based trimming failed: {e}")
        logger.warning("Falling back to atom-based trimming")
        return _trim_by_atoms(protein_positions, ligand_positions, radius, logger)


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