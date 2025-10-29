"""
Structure operations for SO3LR-SF

This module handles structure manipulation operations like trimming and optimization.
All operations return file paths to processed structures for consistent workflow.
"""

import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any, List
from ase import Atoms
from ase.optimize import FIRE, LBFGS
from ase.neighborlist import neighbor_list
from ase.constraints import FixAtoms

from .utils import write_structure, write_opt_structure
from .molecule_loader import load_ase_structure, create_residue_atom_mapping, prepare_mda_universe
from .calculator import So3lrSfCalculator
from .interaction_energy import protein_ligand_interaction
import logging

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
    logger = logging.getLogger(__name__)

    # Set output directory
    if output_dir is None:
        output_dir = protein_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

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
        raise ValueError(f"Unsupported protein file format: {protein_ext}")

    if len(atoms_to_keep) == 0:
        raise ValueError(f"No protein atoms found within {radius} Å of ligand")

    # Create trimmed protein
    trimmed_protein = protein[atoms_to_keep]

    # Generate output filename with trimming method indicator
    # Preserve original format if input is PDB, otherwise use XYZ
    output_ext = protein_ext if protein_ext == '.pdb' else '.xyz'
    trimmed_filename = f"{protein_path.stem}_trimmed_{radius}A_{trimming_method}{output_ext}"
    trimmed_path = output_dir / trimmed_filename

    # Save trimmed protein
    output_path = write_structure(trimmed_protein, trimmed_path)

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

        # First find individual atoms within radius
        atoms_within_radius = set()
        for i, prot_pos in enumerate(protein_positions):
            distances = np.linalg.norm(ligand_positions - prot_pos, axis=1)
            min_distance = np.min(distances)

            if min_distance <= radius:
                atoms_within_radius.add(i)

        # Then identify which residues these atoms belong to
        residues_to_keep = set()
        for residue_id, atom_indices in residue_atom_mapping.items():
            # Check if any atom from this residue is within radius
            if any(atom_idx in atoms_within_radius for atom_idx in atom_indices):
                residues_to_keep.add(residue_id)

        # Finally, include ALL atoms from selected residues
        atoms_to_keep = []
        for residue_id in residues_to_keep:
            atoms_to_keep.extend(residue_atom_mapping[residue_id])

        # Sort atom indices to maintain order
        atoms_to_keep = sorted(set(atoms_to_keep))

        logger.info(f"Residue-based trimming: {len(residues_to_keep)} complete residues selected")
        logger.info(f"Individual atoms within radius: {len(atoms_within_radius)}")
        logger.info(f"Total atoms after including complete residues: {len(atoms_to_keep)}")

        return atoms_to_keep

    except Exception as e:
        logger.warning(f"Residue-based trimming failed: {e}")
        logger.warning("Falling back to atom-based trimming")
        return _trim_by_atoms(protein_positions, ligand_positions, radius, logger)

def perform_trimming(protein_path, ligands_source, radius, trim_lig, output_dir, logger):
    """Handle protein trimming workflow."""
    from .utils import get_ligand_files

    logger.info("=== TRIMMING PHASE ===")

    # Get representative ligand for trimming
    if trim_lig:
        if not Path(trim_lig).exists():
            raise FileNotFoundError(f"Specified trim ligand not found: {trim_lig}")
        representative_ligand = trim_lig
        logger.info(f"Using specified ligand for trimming: {representative_ligand}")
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
    logger = logging.getLogger(__name__)
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


def optimize_structure(
    atoms: Atoms,
    calc: So3lrSfCalculator,
    optimizer: str = 'FIRE',
    fmax: float = 0.01,
    steps: int = 1000,
    output_path: Optional[Union[str, Path]] = None,
    opt_radius: Optional[float] = None,
    n_protein_atoms: Optional[int] = None,
    protein_path: Optional[Union[str, Path]] = None
) -> Tuple[str, Dict[str, Any]]:
    """
    Optimize molecular structure and save as XYZ file.

    This function accepts an ASE Atoms object, optimizes it using the provided
    calculator, and saves the optimized structure as an XYZ file. The structure
    is always saved regardless of convergence status.

    Args:
        atoms: ASE Atoms object to optimize
        calc: So3lrSfCalculator instance to use for optimization
        optimizer: Optimization algorithm ('FIRE' or 'LBFGS')
        fmax: Force convergence criterion in eV/Angstrom (default: 0.01)
        steps: Maximum number of optimization steps (default: 1000)
        output_path: Path for output file
        opt_radius: Optional radius for selective optimization around ligand
        n_protein_atoms: Number of protein atoms (required if opt_radius is used)
        protein_path: Path to original protein file (for residue-level constraints)

    Returns:
        tuple: (output_xyz_path, optimization_info)
               - output_xyz_path: Path to saved optimized XYZ file
               - optimization_info: Dict with convergence details

    Raises:
        ValueError: If optimizer is not recognized or calculator is None
        RuntimeError: If optimization encounters serious errors

    Example:
        >>> calc = So3lrSfCalculator()
        >>> atoms = load_ase_structure("ligand.xyz")
        >>> opt_path, info = optimize_structure(atoms, calc, output_filename="ligand_opt.xyz")
    """
    # Set output directory
    if output_path.is_file():
        # return the path if it's already a file
        return str(output_path), {
            'output_file': output_path,
            'existing_file': True,
            'constraint_info': {
                'constraint_applied': False,
                'constraint_type': None
            }
        }

    initial_positions = atoms.positions.copy()

    # Create and apply constraint if opt_radius is provided
    constraint = None
    if opt_radius is not None and n_protein_atoms is not None:
        constraint = create_optimization_constraint(atoms, n_protein_atoms, opt_radius, protein_path)



    # Use provided So3lrSfCalculator - let it handle fresh initialization
    calc._init_calculator()
    atoms.calc = calc._calculator

    # Calculate initial energy for comparison
    try:
        initial_energy = atoms.get_potential_energy()
    except Exception as e:
        raise RuntimeError(f"Failed to calculate initial energy: {e}")

    if constraint is not None:
        atoms.set_constraint(constraint)
        
    # Choose and configure optimizer
    if optimizer.upper() == 'FIRE':
        opt = FIRE(atoms, logfile=None)
    elif optimizer.upper() == 'LBFGS':
        opt = LBFGS(atoms, logfile=None)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}. Choose 'FIRE' or 'LBFGS'")

    # Run optimization - always save result regardless of convergence
    converged = False
    nsteps = 0
    optimization_error = None

    try:
        opt.run(fmax=fmax, steps=steps)
        converged = opt.converged
        nsteps = opt.nsteps
    except Exception as e:
        optimization_error = str(e)
        print(f"Warning: Optimization encountered issues: {e}")
        print("Saving current structure anyway...")

    # Get final energy (even if optimization didn't converge)
    try:
        final_energy = atoms.get_potential_energy()
    except Exception as e:
        print(f"Warning: Could not calculate final energy: {e}")
        final_energy = initial_energy  # Use initial energy as fallback

    # Calculate structural changes
    displacement = np.linalg.norm(atoms.positions - initial_positions, axis=1)
    max_displacement = np.max(displacement)
    rms_displacement = np.sqrt(np.mean(displacement**2))

    # Save optimized structure as XYZ (always save, even if not converged)
    try:
        output_path_str = write_structure(atoms, output_path)
    except Exception as e:
        raise RuntimeError(f"Failed to save optimized structure: {e}")

    # Prepare constraint information
    constraint_info = {
        'constraint_applied': constraint is not None,
        'constraint_type': type(constraint).__name__ if constraint is not None else None
    }
 
    # Add detailed constraint information if constraint was applied
    if constraint is not None and opt_radius is not None:
        # Extract constraint details
        n_prot = n_protein_atoms
        n_lig = len(atoms) - n_protein_atoms

        # Get flexible and fixed atom counts from constraint
        fixed_atoms = constraint.index if hasattr(constraint, 'index') else []
        n_fixed = len(fixed_atoms)
        n_flexible_protein = n_prot - n_fixed

        constraint_info.update({
            'optimization_radius': opt_radius,
            'total_protein_atoms': n_prot,
            'total_ligand_atoms': n_lig,
            'flexible_protein_atoms': n_flexible_protein,
            'fixed_protein_atoms': n_fixed,
            'ligand_atoms_always_flexible': n_lig,
            'constraint_details': f"{n_flexible_protein}/{n_prot} protein atoms flexible within {opt_radius}Å of ligand"
        })

    # Calculate force information if possible
    initial_forces_max = None
    final_forces_max = None
    try:
        final_forces = atoms.get_forces()
        final_forces_max = np.max(np.linalg.norm(final_forces, axis=1))
    except:
        pass

    # Prepare optimization information
    opt_info = {
        'converged': "yes" if converged else "no",
        'steps': nsteps,
        'steps_taken': nsteps,  # Duplicate for clarity in logs
        'initial_energy': initial_energy,
        'final_energy': final_energy,
        'energy_change': final_energy - initial_energy,
        'max_displacement': max_displacement,
        'rms_displacement': rms_displacement,
        'optimizer': optimizer,
        'fmax_criterion': fmax,
        'max_steps_allowed': steps,
        'output_file': output_path_str,
        'optimization_error': optimization_error,
        'constraint_info': constraint_info,
        'structure_info': {
            'total_atoms': len(atoms),
            'initial_forces_max': initial_forces_max,
            'final_forces_max': final_forces_max
        }
    }

    # Print status
    if converged:
        print(f"Optimization converged in {nsteps} steps")
    else:
        print(f"Optimization did not converge in {nsteps} steps (max: {steps})")

    if optimization_error:
        print(f"Optimization encountered issues: {optimization_error}")

    print(f"Energy change: {final_energy - initial_energy:.6f} eV")
    print(f"Structure saved to: {output_path_str}")

    return output_path_str, opt_info


def extract_ligands(
    multi_structure_file: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    naming_prefix: str = "ligand"
) -> List[str]:
    """
    Extract individual ligands from multi-structure files (SDF, XYZ, PDB with multiple structures).

    This function reads a file containing multiple structures and saves each one
    as a separate XYZ file for individual processing.

    Args:
        multi_structure_file: Path to file containing multiple structures
        output_dir: Directory to save individual ligand files
        naming_prefix: Prefix for output filenames

    Returns:
        List[str]: Paths to individual ligand XYZ files

    Example:
        >>> # Extract ligands from multi-molecule SDF
        >>> ligand_files = extract_ligands("ligands.sdf", output_dir="individual_ligands")
        >>> for lig_file in ligand_files:
        ...     energy = energy_calc_fn(lig_file)
        ...     print(f"{lig_file}: {energy:.3f} eV")
    """
    multi_structure_file = Path(multi_structure_file)

    if output_dir is None:
        output_dir = multi_structure_file.parent / f"{multi_structure_file.stem}_individual"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read all structures
    structures = load_ase_structure(multi_structure_file, index=":")

    output_files = []

    for i, atoms in enumerate(structures):

        # Generate filename
        output_filename = f"{naming_prefix}_{i+1:03d}.xyz"
        output_path = output_dir / output_filename

        # Save structure
        output_file = write_structure(atoms, output_path)
        output_files.append(output_file)

    print(f"Extracted {len(output_files)} structures to {output_dir}")

    return output_files


def optimize_protein(working_protein_path, calc, optimizer, fmax, steps, output_dir,
                    optimization_log, opt_log, logger):
    """Optimize protein structure."""

    logger.info("Optimizing protein...")
    protein_atoms = load_ase_structure(working_protein_path)[0]

    protein_path = Path(working_protein_path)
    output_path = output_dir / f"{protein_path.stem}_opt.xyz"

    optimized_path, opt_info = optimize_structure(
        protein_atoms, calc,
        optimizer=optimizer, fmax=fmax, steps=steps,
        output_path=output_path
    )

    if opt_log and optimization_log is not None:
        optimization_log.append({
            'structure_type': 'protein',
            'structure_name': protein_path.stem,
            'optimization_info': opt_info
        })

    return optimized_path


def process_single_ligand(ligand_file: str, args, calc, working_protein_path: str, output_dir: Path, optimization_log: Optional[List], logger, preloaded_protein_prolif=None):
    """Process a single ligand through the workflow.

    Args:
        ligand_file: Path to ligand structure file
        args: Command line arguments with explainability flags (exp_lig, exp_prot, exp_3d)
        calc: SO3LR-SF calculator instance
        working_protein_path: Path to protein structure file
        output_dir: Output directory for results
        optimization_log: List to store optimization details
        logger: Logging instance
        preloaded_protein_prolif: Pre-loaded protein structure for explainability analysis

    Returns:
        tuple: (result_dict, error_message) where result_dict contains energy and analysis data
    """

    ligand_path = Path(ligand_file)
    ligand_name = ligand_path.stem

    working_ligand_path = ligand_file
    working_complex_path = None

    try:
        # Optimize ligand if requested
        if args.optimize:
            logger.info(f"Optimizing ligand: {ligand_name}")
            ligand_atoms = load_ase_structure(ligand_file)[0]
            working_ligand_path, ligand_opt_info = optimize_structure(
                ligand_atoms, calc,
                optimizer=args.optimizer, fmax=args.fmax, steps=args.steps,
                output_path=output_dir / "opt_ligand" / f"{ligand_name}_opt.xyz"
            )

            if args.opt_log and optimization_log is not None:
                optimization_log.append({
                    'structure_type': 'ligand',
                    'structure_name': ligand_name,
                    'optimization_info': ligand_opt_info
                })

            # Build and optimize complex
            logger.info(f"Building and optimizing complex: {ligand_name}")
            protein_atoms = load_ase_structure(working_protein_path)[0]
            ligand_atoms = load_ase_structure(working_ligand_path)[0]
            complex_atoms = protein_atoms + ligand_atoms

            working_complex_path, complex_opt_info = optimize_structure(
                complex_atoms, calc,
                optimizer=args.optimizer, fmax=args.fmax, steps=args.steps,
                output_path=output_dir / "opt_complexes" / f"{ligand_name}_complex_opt.xyz",
                opt_radius=args.opt_radius, n_protein_atoms=len(protein_atoms)
            )

            if args.opt_log and optimization_log is not None:
                optimization_log.append({
                    'structure_type': 'complex',
                    'structure_name': f"{Path(working_protein_path).stem}_{ligand_name}_complex",
                    'optimization_info': complex_opt_info
                })

            # Extract optimized parts
            optimized_complex_atoms = load_ase_structure(working_complex_path)[0]
            n_protein_atoms = len(protein_atoms)

            optimized_protein_atoms = optimized_complex_atoms[:n_protein_atoms]
            optimized_ligand_atoms = optimized_complex_atoms[n_protein_atoms:n_protein_atoms + len(ligand_atoms)]

            # Save optimized parts
            opt_protein_path = output_dir / f"{Path(working_protein_path).stem}_{ligand_name}_protein_opt.xyz"
            write_structure(optimized_protein_atoms, opt_protein_path)

            opt_ligand_path = write_opt_structure(
                optimized_ligand_atoms, "opt_ligand",
                f"{ligand_name}_from_complex_opt.xyz", output_dir
            )

            working_protein_path = str(opt_protein_path)
            working_ligand_path = str(opt_ligand_path)

        # Calculate interaction energy
        logger.info(f"Calculating interaction energy for: {ligand_name}")

        # Create output paths tuple based on explain modes
        exp_outputs = (
            output_dir / "ligand_exp" / f"{ligand_name}_heatmap.png" if args.exp_lig else None,
            output_dir / "pl_2d_exp" / f"{ligand_name}_protein_interactions.png" if args.exp_prot else None,
            output_dir / "pl_3d_exp" / f"{ligand_name}_3d_visualization.pml" if args.exp_3d else None
        )


        result_from_calc = protein_ligand_interaction(
            working_protein_path, working_ligand_path, calc,
            complex_path=working_complex_path,
            eda=args.eda,
            verbose=False,
            preloaded_protein_prolif=preloaded_protein_prolif,
            exp_outputs=exp_outputs
        )

        # Handle results
        if args.exp_lig or args.exp_prot or args.exp_3d or args.eda:
            interaction_energy, analysis = result_from_calc
        else:
            interaction_energy = result_from_calc
            analysis = {}

        result = {
            'ligand_name': ligand_name,
            'ligand_file': ligand_file,
            'interaction_energy': interaction_energy,
            'binding_energy_kcal_mol': interaction_energy * 23.06,
            'ligand_explainability': analysis
        }

        logger.info(f"  Interaction energy: {interaction_energy:.6f} eV "
                   f"({interaction_energy * 23.06:.2f} kcal/mol)")

        # Component totals are available in analysis['component_totals'] if needed

        # Log EDA component totals
        if analysis.get('interaction_energy_components'):
            logger.info("  EDA Interaction energy components:")
            for comp, total in analysis['interaction_energy_components'].items():
                logger.info(f"    {comp}: {total:.6f} eV")

        return result, None

    except Exception as e:
        error_result = {
            'ligand_name': ligand_name,
            'ligand_file': ligand_file,
            'error': str(e),
            'interaction_energy': float('nan'),
            'binding_energy_kcal_mol': float('nan')
        }
        return error_result, str(e)