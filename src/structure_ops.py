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

from .utils import read_structure, write_structure, validate_structure
from .calculator import So3lrSfCalculator


def trim_structure(
    protein_path: Union[str, Path],
    ligand_path: Union[str, Path],
    radius: float,
    output_dir: Optional[Union[str, Path]] = None
) -> Tuple[str, str]:
    """
    Trim protein structure around ligand within specified radius and save to disk.

    This function creates a trimmed protein structure containing only atoms within
    a specified radius of any ligand atom. The trimmed structure is saved as an XYZ
    file and can be used for more efficient calculations on large protein-ligand systems.

    Args:
        protein_path: Path to protein structure file (PDB, XYZ, etc.)
        ligand_path: Path to ligand structure file
        radius: Radius in Angstroms for trimming around ligand
        output_dir: Directory to save trimmed structures (default: same as protein)

    Returns:
        tuple: (trimmed_protein_path, ligand_path_str)
               - trimmed_protein_path: Path to saved trimmed protein XYZ file
               - ligand_path_str: Path to ligand file (unchanged)

    Raises:
        ValueError: If no protein atoms found within radius or structures are invalid
        FileNotFoundError: If input files don't exist

    Example:
        >>> # Trim protein to 10Å around ligand
        >>> trimmed_prot, lig = trim_structure("protein.pdb", "ligand.sdf", 10.0)
        >>> print(f"Trimmed protein saved to: {trimmed_prot}")
        >>>
        >>> # Use trimmed protein for calculations
        >>> energy = energy_calc_fn(trimmed_prot)
    """
    protein_path = Path(protein_path)
    ligand_path = Path(ligand_path)

    # Set output directory
    if output_dir is None:
        output_dir = protein_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Read structures
    protein = read_structure(protein_path)
    ligand = read_structure(ligand_path)

    # Validate structures
    validate_structure(protein)
    validate_structure(ligand)

    # Get positions
    protein_positions = protein.get_positions()
    ligand_positions = ligand.get_positions()

    # Find protein atoms within radius of any ligand atom
    atoms_to_keep = []

    for i, prot_pos in enumerate(protein_positions):
        # Calculate minimum distance to any ligand atom
        distances = np.linalg.norm(ligand_positions - prot_pos, axis=1)
        min_distance = np.min(distances)

        if min_distance <= radius:
            atoms_to_keep.append(i)

    if len(atoms_to_keep) == 0:
        raise ValueError(f"No protein atoms found within {radius} Å of ligand")

    # Create trimmed protein
    trimmed_protein = protein[atoms_to_keep]

    # Generate output filename
    trimmed_filename = f"{protein_path.stem}_trimmed_{radius}A.xyz"
    trimmed_path = output_dir / trimmed_filename

    # Save trimmed protein
    output_path = write_structure(trimmed_protein, trimmed_path)

    print(f"Trimmed protein ({len(trimmed_protein)} atoms) saved to: {output_path}")

    return output_path


def create_optimization_constraint(
    complex_atoms: Atoms,
    n_protein_atoms: int,
    opt_radius: float
):
    """
    Create constraint for selective optimization based on distance from ligand.

    Args:
        complex_atoms: Combined protein+ligand atoms object
        n_protein_atoms: Number of protein atoms (ligand starts after this index)
        opt_radius: Radius in Angstroms for optimization cutoff

    Returns:
        FixAtoms constraint object or None if no atoms to fix
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

    if len(fixed_protein_idx) > 0:
        print(f"Optimization constraints:")
        print(f"  Flexible protein atoms: {len(flexible_protein)}/{n_prot}")
        print(f"  Fixed protein atoms: {len(fixed_protein_idx)}")
        print(f"  Ligand atoms (always flexible): {n_lig}")

        return FixAtoms(indices=fixed_protein_idx.tolist())
    else:
        print(f"All protein atoms within {opt_radius}Å of ligand - no constraints applied")
        return None


def optimize_structure(
    atoms: Atoms,
    calc: So3lrSfCalculator,
    optimizer: str = 'FIRE',
    fmax: float = 0.01,
    steps: int = 1000,
    output_path: Optional[Union[str, Path]] = None,
    opt_radius: Optional[float] = None,
    n_protein_atoms: Optional[int] = None
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

    Returns:
        tuple: (output_xyz_path, optimization_info)
               - output_xyz_path: Path to saved optimized XYZ file
               - optimization_info: Dict with convergence details

    Raises:
        ValueError: If optimizer is not recognized or calculator is None
        RuntimeError: If optimization encounters serious errors

    Example:
        >>> calc = So3lrSfCalculator()
        >>> atoms = read_structure("ligand.xyz")
        >>> opt_path, info = optimize_structure(atoms, calc, output_filename="ligand_opt.xyz")
    """

    # Validate input atoms
    validate_structure(atoms)

    # Set output directory
    if output_path.is_file():
        # return the path if it's already a file
        return str(output_path), {'output_file': output_path, 'existing_file': True}

    initial_positions = atoms.positions.copy()

    # Create and apply constraint if opt_radius is provided
    constraint = None
    if opt_radius is not None and n_protein_atoms is not None:
        constraint = create_optimization_constraint(atoms, n_protein_atoms, opt_radius)



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
    Extract individual ligands from multi-structure files (SDF, XYZ with multiple frames).

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
    structures = read_structure(multi_structure_file, index=":")

    # Handle case where only one structure is present
    if not isinstance(structures, list):
        structures = [structures]

    output_files = []

    for i, atoms in enumerate(structures):
        validate_structure(atoms)

        # Generate filename
        output_filename = f"{naming_prefix}_{i+1:03d}.xyz"
        output_path = output_dir / output_filename

        # Save structure
        output_file = write_structure(atoms, output_path)
        output_files.append(output_file)

    print(f"Extracted {len(output_files)} structures to {output_dir}")

    return output_files