"""
Structure operations for SO3LR-SF

This module handles structure optimization.
All operations return file paths to processed structures for consistent workflow.
"""

import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, Any, List
from ase import Atoms
import ase.optimize
from ase.constraints import FixAtoms

from .utils import write_structure, write_opt_structure
from .molecule_loader import load_ase_structure
from .calculator import So3lrSfCalculator
from .interaction_energy import protein_ligand_interaction, calculate_strain_energies
from .constraint import create_optimization_constraint
import logging

logger = logging.getLogger(__name__)


def get_optimizer(atoms: Atoms, optimizer: str):
    """
    Get an ASE optimizer instance for the given atoms.

    This function creates and returns an ASE optimizer instance configured for
    the provided atoms object. All optimizers are configured with logfile=None
    to suppress verbose optimization output during calculations.

    Supported optimizers:
        - FIRE: Fast Inertial Relaxation Engine (good for rough optimization)
        - FIRE2: Improved version of FIRE
        - LBFGS: Limited-memory Broyden–Fletcher–Goldfarb–Shanno (memory efficient)
        - BFGS: Broyden–Fletcher–Goldfarb–Shanno (good convergence)
        - BFGSLineSearch: BFGS with line search
        - LBFGSLineSearch: LBFGS with line search
        - GPMin: Gaussian Process Minimization
        - MDMin: Molecular Dynamics Minimization
        - ODE12r: Ordinary Differential Equation solver
        - GoodOldQuasiNewton: Classic quasi-Newton method
        - QuasiNewton: Modern quasi-Newton method

    Args:
        atoms (Atoms): ASE Atoms object to optimize
        optimizer (str): Name of the optimizer (case-insensitive)

    Returns:
        ase.optimize.Optimizer: Configured optimizer instance ready for optimization

    Raises:
        ValueError: If optimizer name is unknown or incompatible with ASE

    Example:
        >>> atoms = load_ase_structure("molecule.xyz")[0]
        >>> opt = get_optimizer(atoms, "LBFGS")
        >>> opt.run(fmax=0.01, steps=1000)
    """
    optimizer_upper = optimizer.upper()

    # Available optimizers in ASE
    if optimizer_upper == 'FIRE':
        return ase.optimize.FIRE(atoms, logfile=None)
    elif optimizer_upper == 'FIRE2':
        return ase.optimize.FIRE2(atoms, logfile=None)
    elif optimizer_upper == 'LBFGS':
        return ase.optimize.LBFGS(atoms, logfile=None)
    elif optimizer_upper == 'BFGS':
        return ase.optimize.BFGS(atoms, logfile=None)
    elif optimizer_upper == 'BFGSLINESEARCH':
        return ase.optimize.BFGSLineSearch(atoms, logfile=None)
    elif optimizer_upper == 'LBFGSLINESEARCH':
        return ase.optimize.LBFGSLineSearch(atoms, logfile=None)
    elif optimizer_upper == 'GPMIN':
        return ase.optimize.GPMin(atoms, logfile=None)
    elif optimizer_upper == 'MDMIN':
        return ase.optimize.MDMin(atoms, logfile=None)
    # elif optimizer_upper == 'CELLAWAREBFGS':
    #     return ase.optimize.CellAwareBFGS(atoms, logfile=None)
    elif optimizer_upper == 'ODE12R':
        return ase.optimize.ODE12r(atoms, logfile=None)
    elif optimizer_upper == 'GOODOLDQUASINEWTON':
        return ase.optimize.GoodOldQuasiNewton(atoms, logfile=None)
    elif optimizer_upper == 'QUASINEWTON':
        return ase.optimize.QuasiNewton(atoms, logfile=None)

    available_optimizers = [
        'FIRE', 'FIRE2', 'LBFGS', 'BFGS', 'BFGSLineSearch', 'LBFGSLineSearch',
        'GPMin', 'MDMin', 'ODE12r', 'GoodOldQuasiNewton', 'QuasiNewton'
    ]

    raise ValueError(f"Unknown optimizer: {optimizer}. Available optimizers: {', '.join(available_optimizers)}")



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
    Optimize molecular structure using ASE optimizers and SO3LR-SF force field.

    This is the core optimization function that handles both unconstrained and
    constrained optimization. For protein-ligand complexes, it can apply radius-based
    constraints where only atoms within a specified distance of the ligand are allowed
    to move during optimization.

    The function always saves the final structure regardless of convergence status,
    ensuring reproducible workflows. Constraint information is tracked and returned
    for use in subsequent strain energy calculations.

    Optimization Process:
        1. Apply constraints if opt_radius is specified
        2. Initialize SO3LR-SF calculator
        3. Run ASE optimization with specified algorithm
        4. Calculate displacement statistics
        5. Save optimized structure to XYZ format
        6. Return file path and detailed optimization information

    Constraint Behavior:
        - When opt_radius is provided: Only protein atoms within opt_radius of any
          ligand atom are optimized, all others are fixed
        - Ligand atoms are always flexible
        - Uses residue-based constraints when possible, falls back to atom-based

    Args:
        atoms (Atoms): ASE Atoms object to optimize (protein+ligand or single molecule)
        calc (So3lrSfCalculator): SO3LR-SF calculator instance for energy/forces
        optimizer (str): ASE optimizer name (default: 'FIRE'). Available: BFGS,
                        BFGSLineSearch, FIRE, FIRE2, GPMin, GoodOldQuasiNewton,
                        LBFGS, LBFGSLineSearch, MDMin, ODE12r, QuasiNewton
        fmax (float): Force convergence criterion in eV/Angstrom (default: 0.01)
        steps (int): Maximum number of optimization steps (default: 1000)
        output_path (Optional[Union[str, Path]]): Output XYZ file path
        opt_radius (Optional[float]): Constraint radius in Angstroms for selective
                                     optimization around ligand
        n_protein_atoms (Optional[int]): Number of protein atoms in complex
                                        (required when opt_radius is used)
        protein_path (Optional[Union[str, Path]]): Path to original protein file
                                                  for residue-based constraints

    Returns:
        Tuple[str, Dict[str, Any]]:
            - str: Path to saved optimized XYZ file
            - Dict: Optimization information containing:
                - 'converged': "yes"/"no" convergence status
                - 'steps'/'steps_taken': Number of optimization steps performed
                - 'max_displacement': Maximum atomic displacement in Angstroms
                - 'rms_displacement': RMS atomic displacement in Angstroms
                - 'optimizer': Optimizer algorithm used
                - 'fmax_criterion': Force convergence threshold
                - 'max_steps_allowed': Maximum steps allowed
                - 'output_file': Path to saved structure
                - 'optimization_error': Error message if optimization failed
                - 'constraint_info': Detailed constraint information including:
                    - 'constraint_applied': Boolean if constraints were used
                    - 'optimization_radius': Radius used for constraints
                    - 'flexible_protein_atoms': Number of moveable protein atoms
                    - 'constraint': ASE constraint object for strain calculations

    Raises:
        ValueError: If optimizer is not recognized or calculator is None
        RuntimeError: If structure saving fails or serious optimization errors occur

    Examples:
        >>> # Unconstrained optimization
        >>> calc = So3lrSfCalculator()
        >>> atoms = load_ase_structure("ligand.xyz")[0]
        >>> opt_path, info = optimize_structure(atoms, calc, output_path="ligand_opt.xyz")

        >>> # Constrained optimization of protein-ligand complex
        >>> complex_atoms = protein_atoms + ligand_atoms
        >>> opt_path, info = optimize_structure(
        ...     complex_atoms, calc, opt_radius=4.0, n_protein_atoms=len(protein_atoms),
        ...     output_path="complex_opt.xyz"
        ... )

        >>> # Check optimization results
        >>> if info['converged'] == 'yes':
        ...     print(f"Converged in {info['steps']} steps")
        >>> print(f"Max displacement: {info['max_displacement']:.3f} Å")
    """
    initial_positions = atoms.positions.copy()

    # Create and apply constraint if opt_radius is provided
    constraint = None
    if opt_radius is not None and n_protein_atoms is not None:
        logger.info(f"Applying optimization constraints with radius: {opt_radius} Å")
        constraint = create_optimization_constraint(atoms, n_protein_atoms, opt_radius, protein_path)
        
        # Extract constraint details
        n_prot = n_protein_atoms
        n_lig = len(atoms) - n_protein_atoms

        # Get flexible and fixed atom counts from constraint
        fixed_atoms = constraint.index if hasattr(constraint, 'index') else []
        n_fixed = len(fixed_atoms)
        n_flexible_protein = n_prot - n_fixed
        
        constraint_info = {
            'constraint_applied': constraint is not None,
            'constraint_type': type(constraint).__name__ if constraint is not None else None,
            'optimization_radius': opt_radius,
            'total_protein_atoms': n_prot,
            'total_ligand_atoms': n_lig,
            'flexible_protein_atoms': n_flexible_protein,
            'fixed_protein_atoms': n_fixed,
            'ligand_atoms_always_flexible': n_lig,
            'constraint_details': f"{n_flexible_protein}/{n_prot} protein atoms flexible within {opt_radius}Å of ligand",
            'constraint': constraint
        }
    else:
        constraint_info = None
    
    # Set output directory
    if output_path.is_file():
        # return the path if it's already a file
        logger.info(f"Output path is already a file: {output_path}, skipping optimization.")
        return str(output_path), {
            'output_file': output_path,
            'existing_file': True,
            'constraint_info': constraint_info
        }

    # Use provided So3lrSfCalculator - let it handle fresh initialization
    calc._init_calculator()
    atoms.calc = calc._calculator

    if constraint is not None:
        atoms.set_constraint(constraint)

    # Choose and configure optimizer
    opt = get_optimizer(atoms, optimizer)

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
        logger.error(f"Optimization encountered issues: {e}")
        logger.warning("Saving current structure anyway...")

    # Calculate structural changes
    displacement = np.linalg.norm(atoms.positions - initial_positions, axis=1)
    max_displacement = np.max(displacement)
    rms_displacement = np.sqrt(np.mean(displacement**2))

    # Save optimized structure as XYZ (always save, even if not converged)
    try:
        output_path_str = write_structure(atoms, output_path)
    except Exception as e:
        raise RuntimeError(f"Failed to save optimized structure: {e}")

    # Prepare optimization information
    opt_info = {
        'converged': "yes" if converged else "no",
        'steps': nsteps,
        'steps_taken': nsteps,  # Duplicate for clarity in logs
        'max_displacement': max_displacement,
        'rms_displacement': rms_displacement,
        'optimizer': optimizer,
        'fmax_criterion': fmax,
        'max_steps_allowed': steps,
        'output_file': output_path_str,
        'optimization_error': optimization_error,
        'constraint_info': constraint_info
    }

    if converged:
        logger.info(f"Optimization converged in {nsteps} steps (max displacement: {max_displacement:.6f} Å)")
    else:
        logger.info(f"Optimization did not converge in {nsteps} steps (max: {steps}, max displacement: {max_displacement:.6f} Å)")

    if optimization_error:
        logger.warning(f"Optimization encountered issues: {optimization_error}")

    logger.info(f"Structure saved to: {output_path_str}")

    return output_path_str, opt_info

def constrained_optimization(ligand_file: str, ligand_name: str, working_protein_path: str,
                                  output_dir: Path, calc, args, optimization_log: Optional[List], logger):
    """
    Perform constrained optimization of protein-ligand complex for strain calculations.

    This function implements the default optimization workflow used in both 'no-strain'
    and strain-based optimization modes. It builds a protein-ligand complex from
    separate structures, optimizes the complex with radius-based constraints, and
    extracts optimized components for energy calculations.

    When strain modes are requested ('strain' or 'strain-prot'), this function also
    triggers free optimization of individual components to calculate strain energies
    that are added to the base interaction energy.

    Workflow:
        1. Load and combine protein + ligand structures into complex
        2. Apply constrained optimization (only binding pocket atoms move)
        3. Extract optimized protein and ligand from optimized complex
        4. Save separated components for interaction energy calculation
        5. If strain mode: perform additional free optimizations
        6. Return paths to all optimized structures

    Constraint Details:
        - Ligand atoms: Always flexible during optimization
        - Protein atoms: Only those within opt_radius of ligand can move
        - Constraint information preserved for consistent strain calculations

    Args:
        ligand_file (str): Path to input ligand structure file
        ligand_name (str): Name identifier for ligand (used in output filenames)
        working_protein_path (str): Path to input protein structure file
        output_dir (Path): Base directory for saving optimization outputs
        calc (So3lrSfCalculator): SO3LR-SF calculator for energy/force computation
        args: Argument object containing optimization parameters:
            - optimization_mode (str): 'no-strain', 'strain', or 'strain-prot'
            - opt_radius (float): Constraint radius in Angstroms
            - optimizer (str): ASE optimizer algorithm name
            - fmax (float): Force convergence criterion
            - steps (int): Maximum optimization steps
            - opt_log (bool): Whether to log optimization details
            - calculate_protein_strain (bool): Include protein strain calculation
        optimization_log (Optional[List]): List to store optimization metadata
        logger: Python logger for progress reporting

    Returns:
        Tuple[str, str, str, Optional[str], Optional[str]]:
            - opt_protein_path: Path to optimized protein component
            - opt_ligand_path: Path to optimized ligand component
            - optimized_complex_path: Path to optimized complex
            - optimized_free_ligand_path: Path to free ligand (strain modes only)
            - optimized_free_protein_path: Path to free protein (strain modes only)

    File Organization:
        output_dir/
        ├── constrained_opt_complexes/
        │   └── {ligand_name}_complex_constrained_opt.xyz
        ├── constrained_opt_components/
        │   ├── {protein_stem}_{ligand_name}_constrained_opt.xyz
        │   └── {ligand_name}_constrained_opt.xyz
        ├── free_opt_ligands/  (strain modes only)
        │   └── {ligand_name}_free_opt.xyz
        └── free_opt_protein/  (strain-prot mode only)
            └── {protein_stem}_free_opt.xyz

    Notes:
        - Always performs constrained optimization as the base calculation
        - Strain energies are additional corrections added to base interaction energy
        - Constraint objects are preserved and passed to strain optimization
        - Function logs detailed progress and optimization statistics
        - Components are extracted from complex to ensure consistency

    Example:
        >>> output_dir = Path("optimization_results")
        >>> prot_path, lig_path, complex_path, free_lig, free_prot = constrained_optimization(
        ...     "ligand.sdf", "aspirin", "protein.pdb", output_dir, calc, args, [], logger
        ... )
        >>> # Use returned paths for interaction energy calculation
    """
    logger.info(f"=== OPTIMIZATION MODE: {args.optimization_mode.upper()} (radius: {args.opt_radius} Å) ===")

    # Build complex from original structures
    protein_atoms = load_ase_structure(working_protein_path)[0]
    ligand_atoms = load_ase_structure(ligand_file)[0]
    complex_atoms = protein_atoms + ligand_atoms

    logger.debug(f"Complex built: {len(protein_atoms)} protein + {len(ligand_atoms)} ligand = {len(complex_atoms)} total atoms")

    # Optimize the complex with constrained optimization
    opt_complex_path = output_dir / "constrained_opt_complexes" / f"{ligand_name}_complex_constrained_opt.xyz"
    opt_complex_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Optimizing complex with ligand and binding pocket constraints...")

    optimized_complex_path, complex_opt_info = optimize_structure(
        complex_atoms, calc,
        optimizer=args.optimizer, fmax=args.fmax, steps=args.steps,
        output_path=opt_complex_path,
        opt_radius=args.opt_radius, n_protein_atoms=len(protein_atoms),
        protein_path=working_protein_path
    )

    if args.opt_log and optimization_log is not None:
        optimization_log.append({
            'structure_type': 'complex_constrained',
            'structure_name': f"{Path(working_protein_path).stem}_{ligand_name}_complex_constrained",
            'optimization_info': complex_opt_info
        })

    # Extract optimized protein and ligand from the optimized complex
    optimized_complex_atoms = load_ase_structure(optimized_complex_path)[0]
    n_protein_atoms = len(protein_atoms)

    optimized_protein_atoms = optimized_complex_atoms[:n_protein_atoms]
    optimized_ligand_atoms = optimized_complex_atoms[n_protein_atoms:n_protein_atoms + len(ligand_atoms)]


    # Save separated optimized components for energy calculations
    components_dir = output_dir / "constrained_opt_components"
    components_dir.mkdir(parents=True, exist_ok=True)

    opt_protein_path = components_dir / f"{Path(working_protein_path).stem}_{ligand_name}_constrained_opt.xyz"
    opt_ligand_path = components_dir / f"{ligand_name}_constrained_opt.xyz"

    write_structure(optimized_protein_atoms, opt_protein_path)
    write_structure(optimized_ligand_atoms, opt_ligand_path)

    logger.debug(f"Complex components extracted and saved (protein: {opt_protein_path}, ligand: {opt_ligand_path}, complex: {optimized_complex_path})")

    optimized_free_ligand_path = None
    optimized_free_protein_path = None

    if args.optimization_mode in ['strain', 'strain-prot']:
        logger.info(f"Computing strain energies ({args.optimization_mode})...")

        # Get flexible protein atom indices from complex optimization info
        constraint = None
        constraint_info = complex_opt_info.get('constraint_info', {})

        if constraint_info and 'constraint' in constraint_info:
            constraint = constraint_info.get('constraint')
        else:
            logger.warning("No constraint information available for strain calculation, proceeding without constraints.")

        optimized_free_ligand_path, optimized_free_protein_path = strain_optimization(
            ligand_file, ligand_name, working_protein_path, output_dir, calc, args, optimization_log, logger,
            constraint=constraint
        )
    logger.info(f"Strain optimization complete - ready for energy calculations")
    logger.debug(f"=== CONSTRAINED OPTIMIZATION COMPLETED ===")
    if optimized_free_ligand_path:
        logger.debug(f"  Free ligand: {optimized_free_ligand_path}")
    if optimized_free_protein_path:
        logger.debug(f"  Free protein: {optimized_free_protein_path}")

    return str(opt_protein_path), str(opt_ligand_path), optimized_complex_path, optimized_free_ligand_path, optimized_free_protein_path


def strain_optimization(ligand_file: str, ligand_name: str, working_protein_path: str,
                       output_dir: Path, calc, args, optimization_log: Optional[List], logger, constraint: Optional[set[FixAtoms]]):
    """
    Perform free optimization of individual components for strain energy calculation.

    This function is called by constrained_optimization() when strain modes are requested
    ('strain' or 'strain-prot'). It optimizes ligand and protein structures in their
    free (unbound) conformations to calculate strain energies - the energetic cost of
    deforming molecules from their optimal free conformations to bound conformations.

    Strain Energy Concept:
        - Ligand strain = E_ligand_in_complex - E_ligand_free_optimized
        - Protein strain = E_protein_in_complex - E_protein_free_optimized (with constraints)
        - Total binding energy = Interaction energy + Ligand strain + Protein strain

    Optimization Strategy:
        1. Ligand: Unconstrained optimization (represents free solution state)
        2. Protein: Constrained optimization using same constraints as complex
           (represents protein with same binding site flexibility)

    Args:
        ligand_file (str): Path to original ligand structure file
        ligand_name (str): Ligand identifier for output file naming
        working_protein_path (str): Path to original protein structure file
        output_dir (Path): Base directory for optimization outputs
        calc (So3lrSfCalculator): SO3LR-SF calculator for energy/force computation
        args: Arguments object containing:
            - optimization_mode (str): 'strain' (ligand only) or 'strain-prot' (both)
            - optimizer (str): ASE optimizer algorithm
            - fmax (float): Force convergence criterion
            - steps (int): Maximum optimization steps
            - opt_log (bool): Enable optimization logging
            - calculate_protein_strain (bool): Whether to optimize protein
        optimization_log (Optional[List]): List for storing optimization metadata
        logger: Python logger for progress reporting
        constraint (Optional[set[int]]): ASE constraint object from complex optimization
                                        to apply consistent constraints to protein

    Returns:
        Tuple[str, Optional[str]]:
            - optimized_free_ligand_path: Path to free-optimized ligand structure
            - optimized_free_protein_path: Path to free-optimized protein (if requested)

    File Outputs:
        - free_opt_ligands/{ligand_name}_free_opt.xyz: Free ligand structure
        - free_opt_protein/{protein_stem}_free_opt.xyz: Free protein structure (optional)

    Notes:
        - Ligand optimization is always performed (required for any strain calculation)
        - Protein optimization only when calculate_protein_strain=True
        - Protein uses same constraints as complex to maintain binding site flexibility
        - Results enable strain energy calculation in calculate_strain_energies()
        - Function is designed to be called after constrained_optimization()

    Example:
        >>> # Called from constrained_optimization() for strain modes
        >>> constraint = complex_opt_info['constraint_info']['constraint']
        >>> free_lig_path, free_prot_path = strain_optimization(
        ...     "ligand.sdf", "aspirin", "protein.pdb", output_dir, calc,
        ...     args, optimization_log, logger, constraint
        ... )
    """
    logger.info(f"=== STRAIN OPTIMIZATION ({args.optimization_mode}) ===")

    # Step 1: Optimize ligand alone (required for strain calculation)
    logger.debug(f"Optimizing free ligand for strain calculation: {ligand_name}")
    ligand_atoms = load_ase_structure(ligand_file)[0]
    free_ligand_path = output_dir / "free_opt_ligands" / f"{ligand_name}_free_opt.xyz"
    free_ligand_path.parent.mkdir(parents=True, exist_ok=True)

    optimized_free_ligand_path, ligand_opt_info = optimize_structure(
        ligand_atoms, calc,
        optimizer=args.optimizer, fmax=args.fmax, steps=args.steps, opt_radius=None,
        output_path=free_ligand_path
    )

    if args.opt_log and optimization_log is not None:
        optimization_log.append({
            'ligand_strain_info': ligand_opt_info
        })
    logger.info(f"Free ligand optimization completed: {optimized_free_ligand_path}")


    # Step 2: Optimize protein alone (if protein strain requested)
    optimized_free_protein_path = None
    if hasattr(args, 'calculate_protein_strain') and args.calculate_protein_strain:
        logger.debug(f"Optimizing free protein with same constraints for strain calculation ...")
        protein_stem = Path(working_protein_path).stem
        free_protein_path = output_dir / "free_opt_protein" / f"{protein_stem}_free_opt.xyz"

        if free_protein_path.exists():
            logger.debug(f"Using existing free optimized protein: {free_protein_path}")
            optimized_free_protein_path = str(free_protein_path)
        else:
            logger.debug(f"Loading and optimizing protein structure: {protein_stem}")
            free_protein_path.parent.mkdir(parents=True, exist_ok=True)
            protein_atoms = load_ase_structure(working_protein_path)[0]
            logger.debug(f"Protein loaded: {len(protein_atoms)} atoms")

            if constraint is not None:
                logger.debug(f"Applying same constraints as complex optimization to protein")
                protein_atoms.set_constraint(constraint)
            else:
                logger.warning("No constraint available for protein strain optimization")

            optimized_free_protein_path, protein_opt_info = optimize_structure(
                protein_atoms, calc,
                optimizer=args.optimizer, fmax=args.fmax, steps=args.steps,
                output_path=free_protein_path
            )

            if args.opt_log and optimization_log is not None:
                optimization_log.append({
                    'structure_type': 'protein_free_for_strain',
                    'structure_name': protein_stem,
                    'optimization_info': protein_opt_info
                })
            logger.info(f"Free protein optimization completed: {optimized_free_protein_path}")
    else:
        logger.debug("Protein strain calculation not requested, skipping protein optimization")

    logger.info(f"Strain optimization completed - free structures optimized")
    logger.debug(f"Free ligand: {optimized_free_ligand_path}")
    logger.debug(f"Free protein: {optimized_free_protein_path}")

    return optimized_free_ligand_path, optimized_free_protein_path


def process_single_ligand(ligand_file: str, args, calc, working_protein_path: str, output_dir: Path, optimization_log: Optional[List], logger, preloaded_protein_prolif=None, charges=(0, 0, 0)):
    """
    Complete workflow for processing a single ligand with protein-ligand interaction analysis.

    This is the main entry point for ligand processing that orchestrates the entire
    computational workflow including optimization, interaction energy calculation,
    strain energy analysis, and explainability features. The function handles both
    traditional interaction energy calculations and advanced strain-based analysis.

    Workflow Overview:
        1. Structure optimization (if requested via args.optimize)
           - Constrained optimization of protein-ligand complex
           - Optional free optimization for strain energy calculation
        2. Protein-ligand interaction energy calculation
        3. Strain energy calculation and integration (strain modes only)
        4. Explainability analysis (if requested)
        5. Energy decomposition analysis (if requested)

    Optimization Modes:
        - No optimization: Use original structures directly
        - 'no-strain': Constrained optimization only (traditional approach)
        - 'strain': Add ligand strain energy to interaction energy
        - 'strain-prot': Add both ligand and protein strain energies

    Energy Components:
        - Base interaction energy: E_complex - E_protein - E_ligand (optimized)
        - Ligand strain: E_ligand_in_complex - E_ligand_free
        - Protein strain: E_protein_in_complex - E_protein_free
        - Final energy: Base + strain corrections

    Args:
        ligand_file (str): Path to ligand structure file (SDF, MOL2, XYZ, etc.)
        args: Arguments object containing workflow parameters:
            - optimize (bool): Whether to perform structural optimization
            - optimization_mode (str): 'no-strain', 'strain', or 'strain-prot'
            - opt_radius (float): Constraint radius for optimization
            - optimizer, fmax, steps: Optimization parameters
            - exp_lig (bool): Generate ligand explainability heatmap
            - exp_prot (bool): Generate 2D protein-ligand interaction diagram
            - exp_3d (bool): Generate 3D PyMOL visualization script
            - eda (bool): Perform energy decomposition analysis
            - opt_log (bool): Log detailed optimization information
            - calculate_protein_strain (bool): Include protein strain calculation
            - verbose (bool): Enable verbose output
        calc (So3lrSfCalculator): SO3LR-SF calculator for energy/force computation
        working_protein_path (str): Path to protein structure file
        output_dir (Path): Base directory for all output files
        optimization_log (Optional[List]): List to collect optimization metadata
        logger: Python logger for progress reporting
        preloaded_protein_prolif (Optional): Pre-loaded protein for explainability
                                            analysis (performance optimization)

    Returns:
        Tuple[Dict[str, Any], Optional[str]]:
            - result_dict: Dictionary containing:
                - 'ligand_name': Ligand identifier
                - 'ligand_file': Path to original ligand file
                - 'interaction_energy': Final interaction energy (eV)
                - 'base_interaction_energy': Base interaction energy before strain
                - 'ligand_strain_energy': Ligand strain correction (eV)
                - 'protein_strain_energy': Protein strain correction (eV)
                - 'binding_energy_kcal_mol': Final energy in kcal/mol
                - 'ligand_explainability': Analysis results and components
            - error_message: None if successful, error string if failed

    Output Files Generated:
        Optimization outputs:
            - constrained_opt_complexes/{ligand}_complex_constrained_opt.xyz
            - constrained_opt_components/{protein}_{ligand}_constrained_opt.xyz
            - constrained_opt_components/{ligand}_constrained_opt.xyz
            - free_opt_ligands/{ligand}_free_opt.xyz (strain modes)
            - free_opt_protein/{protein}_free_opt.xyz (strain-prot mode)

        Explainability outputs (if requested):
            - ligand_exp/{ligand}_heatmap.png (per-atom contribution heatmap)
            - pl_2d_exp/{ligand}_protein_interactions.png (2D interaction diagram)
            - pl_3d_exp/{ligand}_3d_visualization.pml (PyMOL visualization script)

    Error Handling:
        - Returns error information in result dict if any step fails
        - Preserves partial results when possible
        - Logs detailed error information for debugging

    Examples:
        >>> # Basic interaction energy calculation
        >>> result, error = process_single_ligand(
        ...     "ligand.sdf", args, calc, "protein.pdb", output_dir, [], logger
        ... )
        >>> print(f"Interaction energy: {result['interaction_energy']:.3f} eV")

        >>> # With strain analysis and explainability
        >>> args.optimization_mode = 'strain'
        >>> args.exp_lig = True
        >>> args.eda = True
        >>> result, error = process_single_ligand(
        ...     "ligand.sdf", args, calc, "protein.pdb", output_dir, [], logger
        ... )
        >>> print(f"Base energy: {result['base_interaction_energy']:.3f} eV")
        >>> print(f"Ligand strain: {result['ligand_strain_energy']:.3f} eV")
        >>> print(f"Final energy: {result['interaction_energy']:.3f} eV")
    """

    ligand_path = Path(ligand_file)
    ligand_name = ligand_path.stem

    working_ligand_path = ligand_file
    working_complex_path = None

    try:
        # Check if optimization is requested
        if args.optimize:
            working_protein_path, working_ligand_path, working_complex_path, free_ligand_path, free_protein_path = constrained_optimization(
                ligand_file, ligand_name, working_protein_path, output_dir, calc, args, optimization_log, logger
            )
        else:
            # No optimization - use original structures
            working_ligand_path = ligand_file
            working_complex_path = None
            free_ligand_path = None
            free_protein_path = None

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
            verbose=getattr(args, 'verbose', False),
            preloaded_protein_prolif=preloaded_protein_prolif,
            exp_outputs=exp_outputs,
            charges=charges
        )

        # Handle results
        if args.exp_lig or args.exp_prot or args.exp_3d or args.eda:
            interaction_energy, analysis = result_from_calc
        else:
            interaction_energy = result_from_calc
            analysis = {}

        # Calculate strain energies if in strain mode and optimization was performed
        final_interaction_energy = interaction_energy
        ligand_strain = 0.0
        protein_strain = 0.0
        strain_energies = {}

        if args.optimize and hasattr(args, 'optimization_mode') and args.optimization_mode in ['strain', 'strain-prot']:
            # Only calculate strain energies if we have the required free structure paths
            if free_ligand_path is not None:
                strain_energies = calculate_strain_energies(
                    working_ligand_path, working_protein_path,
                    free_ligand_path, free_protein_path,
                    calc, calculate_protein_strain=getattr(args, 'calculate_protein_strain', False),
                    logger=logger
                )

                # Add strain energies to the final interaction energy
                ligand_strain = strain_energies.get('ligand_strain', 0.0)
                protein_strain = strain_energies.get('protein_strain', 0.0)

                final_interaction_energy = interaction_energy + ligand_strain + protein_strain

                logger.info(f"  Base interaction energy: {interaction_energy:.6f} eV")
                if ligand_strain != 0.0:
                    logger.info(f"  Ligand strain energy: {ligand_strain:.6f} eV ({ligand_strain * 23.06:.2f} kcal/mol)")
                if protein_strain != 0.0:
                    logger.info(f"  Protein strain energy: {protein_strain:.6f} eV ({protein_strain * 23.06:.2f} kcal/mol)")
                logger.info(f"  Final interaction energy (including strain): {final_interaction_energy:.6f} eV "
                           f"({final_interaction_energy * 23.06:.2f} kcal/mol)")

        result = {
            'ligand_name': ligand_name,
            'ligand_file': ligand_file,
            'interaction_energy': final_interaction_energy,
            'base_interaction_energy': interaction_energy,
            'ligand_strain_energy': ligand_strain,
            'protein_strain_energy': protein_strain,
            'binding_energy_kcal_mol': final_interaction_energy * 23.06,
            'ligand_explainability': analysis
        }

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