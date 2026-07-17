"""
Structure constraint operations for SO3LR-SF

This module handles optimization constraints for selective structure optimization.
All functions work with ASE Atoms objects and return constraint information.
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional, List
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.constraints import FixAtoms
import logging

from .molecule_loader import create_residue_atom_mapping, prepare_mda_universe

logger = logging.getLogger(__name__)


def _constrain_by_atoms(complex_atoms: Atoms, n_protein_atoms: int, opt_radius: float, logger) -> List[int]:
    """
    Find protein atoms to fix using atom-level constraints.

    Args:
        complex_atoms: Combined protein+ligand atoms object
        n_protein_atoms: Number of protein atoms
        opt_radius: Radius in Angstroms for optimization cutoff
        logger: Logger instance

    Returns:
        List of protein atom indices to fix
    """
    n_prot = n_protein_atoms
    n_lig = len(complex_atoms) - n_protein_atoms

    # Define atom indices
    protein_idx = np.arange(n_prot)
    ligand_idx = np.arange(n_prot, n_prot + n_lig)

    # Get all pairs (i -> central atom, j -> neighbor) and distances
    i, j, d = neighbor_list('ijd', complex_atoms, opt_radius)

    # Find protein atoms within cutoff of any ligand atom
    mask = np.isin(i, ligand_idx) & (j < n_prot)
    flexible_protein = np.unique(j[mask])

    # Fixed atoms are protein atoms NOT within the flexible region
    fixed_protein_idx = np.setdiff1d(protein_idx, flexible_protein)

    logger.debug(f"Atom-based constraints: {len(flexible_protein)} flexible, {len(fixed_protein_idx)} fixed")
    return fixed_protein_idx.tolist()


def _constrain_by_residues(complex_atoms: Atoms, n_protein_atoms: int, opt_radius: float,
                          protein_path: Path, logger) -> List[int]:
    """
    Find protein atoms to fix using residue-level constraints.

    Args:
        complex_atoms: Combined protein+ligand atoms object
        n_protein_atoms: Number of protein atoms
        opt_radius: Radius in Angstroms for optimization cutoff
        protein_path: Path to PDB protein file
        logger: Logger instance

    Returns:
        List of protein atom indices to fix (complete residues)
    """
    try:
        # Get residue information from PDB
        universe = prepare_mda_universe(protein_path)
        residue_atom_mapping = create_residue_atom_mapping(universe)

        # First find individual atoms within radius using existing logic
        n_prot = n_protein_atoms
        n_lig = len(complex_atoms) - n_protein_atoms
        ligand_idx = np.arange(n_prot, n_prot + n_lig)

        # Get all pairs and find flexible atoms
        i, j, d = neighbor_list('ijd', complex_atoms, opt_radius)
        mask = np.isin(i, ligand_idx) & (j < n_prot)
        flexible_atoms = set(np.unique(j[mask]))

        # Determine which residues have any flexible atoms
        flexible_residues = set()
        for residue_id, atom_indices in residue_atom_mapping.items():
            if any(atom_idx in flexible_atoms for atom_idx in atom_indices):
                flexible_residues.add(residue_id)

        # All atoms from non-flexible residues should be fixed
        fixed_atoms = []
        for residue_id, atom_indices in residue_atom_mapping.items():
            if residue_id not in flexible_residues:
                fixed_atoms.extend(atom_indices)

        # Sort for consistency
        fixed_atoms = sorted(set(fixed_atoms))

        logger.info(f"Residue-based constraints: {len(flexible_residues)} flexible residues")
        logger.info(f"Individual atoms within radius: {len(flexible_atoms)}")
        logger.info(f"Total atoms after including complete residues: {len(flexible_atoms)} flexible, {len(fixed_atoms)} fixed")

        return fixed_atoms

    except Exception as e:
        logger.warning(f"Residue-based constraints failed: {e}")
        logger.warning("Falling back to atom-based constraints")
        return _constrain_by_atoms(complex_atoms, n_protein_atoms, opt_radius, logger)


def create_optimization_constraint(
    complex_atoms: Atoms,
    n_protein_atoms: int,
    opt_radius: float,
    protein_path: Optional[Union[str, Path]] = None
):
    """
    Create constraint for selective optimization based on distance from ligand.

    Uses residue-level constraints for PDB files and atom-level constraints for other formats.

    Args:
        complex_atoms: Combined protein+ligand atoms object
        n_protein_atoms: Number of protein atoms (ligand starts after this index)
        opt_radius: Radius in Angstroms for optimization cutoff
        protein_path: Path to original protein file (for format detection)

    Returns:
        FixAtoms constraint object or None if no atoms to fix
    """
    n_prot = n_protein_atoms
    n_lig = len(complex_atoms) - n_protein_atoms

    # Determine constraint strategy based on protein file format
    if protein_path is not None:
        protein_path = Path(protein_path)
        protein_ext = protein_path.suffix.lower()

        if protein_ext == '.pdb':
            # Residue-based constraints for PDB files
            logger.info(f"Using residue-level constraints for PDB file: {protein_path.name}")
            logger.info("Complete residues will be flexible if any atom is within the cutoff radius")
            fixed_atoms = _constrain_by_residues(complex_atoms, n_protein_atoms, opt_radius, protein_path, logger)
            constraint_method = "residue"
        else:
            # Atom-based constraints for non-PDB files
            logger.warning(f"Using atom-level constraints for {protein_ext.upper()} file: {protein_path.name}")
            logger.warning("Individual atoms will be constrained - residues may be split!")
            logger.warning("For complete residue constraints, use PDB format input files")
            fixed_atoms = _constrain_by_atoms(complex_atoms, n_protein_atoms, opt_radius, logger)
            constraint_method = "atom"
    else:
        # Default to atom-based constraints when no protein path provided
        logger.info("No protein path provided - using atom-level constraints")
        fixed_atoms = _constrain_by_atoms(complex_atoms, n_protein_atoms, opt_radius, logger)
        constraint_method = "atom"

    if len(fixed_atoms) > 0:
        flexible_count = n_prot - len(fixed_atoms)
        logger.info(f"Optimization constraints ({constraint_method}-based):")
        logger.info(f"  Flexible protein atoms: {flexible_count}/{n_prot}")
        logger.info(f"  Fixed protein atoms: {len(fixed_atoms)}")
        logger.info(f"  Ligand atoms (always flexible): {n_lig}")

        return FixAtoms(indices=fixed_atoms)
    else:
        logger.info(f"All protein atoms within {opt_radius}Å of ligand - no constraints applied")
        return None