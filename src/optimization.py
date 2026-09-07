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

from .utils import write_structure, write_opt_structure
from .molecule_loader import load_ase_structure
from .calculator import So3lrSfCalculator
from .interaction_energy import protein_ligand_interaction
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
    for use in subsequent steps.

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
                    - 'constraint': ASE constraint object

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
    # For optimization, we always need the MLFF calculator since JAX-MD doesn't support forces
    calc._init_so3lr_calculator()
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
    Perform constrained optimization of a protein-ligand complex.

    This function implements the optimization workflow. It builds a protein-ligand
    complex from separate structures, optimizes the complex with radius-based
    constraints, and extracts the optimized components for energy calculations.

    Workflow:
        1. Load and combine protein + ligand structures into complex
        2. Apply constrained optimization (only binding pocket atoms move)
        3. Extract optimized protein and ligand from optimized complex
        4. Save separated components for interaction energy calculation
        5. Return paths to the optimized structures

    Constraint Details:
        - Ligand atoms: Always flexible during optimization
        - Protein atoms: Only those within opt_radius of ligand can move

    Args:
        ligand_file (str): Path to input ligand structure file
        ligand_name (str): Name identifier for ligand (used in output filenames)
        working_protein_path (str): Path to input protein structure file
        output_dir (Path): Base directory for saving optimization outputs
        calc (So3lrSfCalculator): SO3LR-SF calculator for energy/force computation
        args: Argument object containing optimization parameters:
            - opt_radius (float): Constraint radius in Angstroms
            - optimizer (str): ASE optimizer algorithm name
            - fmax (float): Force convergence criterion
            - steps (int): Maximum optimization steps
            - opt_log (bool): Whether to log optimization details
        optimization_log (Optional[List]): List to store optimization metadata
        logger: Python logger for progress reporting

    Returns:
        Tuple[str, str, str]:
            - opt_protein_path: Path to optimized protein component
            - opt_ligand_path: Path to optimized ligand component
            - optimized_complex_path: Path to optimized complex

    File Organization:
        output_dir/
        ├── constrained_opt_complexes/
        │   └── {ligand_name}_complex_constrained_opt.xyz
        └── constrained_opt_components/
            ├── {protein_stem}_{ligand_name}_constrained_opt.xyz
            └── {ligand_name}_constrained_opt.xyz

    Notes:
        - Function logs detailed progress and optimization statistics
        - Components are extracted from complex to ensure consistency

    Example:
        >>> output_dir = Path("optimization_results")
        >>> prot_path, lig_path, complex_path = constrained_optimization(
        ...     "ligand.sdf", "aspirin", "protein.pdb", output_dir, calc, args, [], logger
        ... )
        >>> # Use returned paths for interaction energy calculation
    """
    logger.info(f"=== CONSTRAINED OPTIMIZATION (radius: {args.opt_radius} Å) ===")

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

    logger.debug(f"=== CONSTRAINED OPTIMIZATION COMPLETED ===")

    return str(opt_protein_path), str(opt_ligand_path), optimized_complex_path


def process_single_ligand(ligand_file: str, args, calc, working_protein_path: str, output_dir: Path, optimization_log: Optional[List], logger, preloaded_protein_prolif=None, charges=(0, 0, 0)):
    """
    Complete workflow for processing a single ligand with protein-ligand interaction analysis.

    This is the main entry point for ligand processing that orchestrates the entire
    computational workflow including optimization, interaction energy calculation,
    and explainability features.

    Workflow Overview:
        1. Structure optimization (if requested via args.optimize)
           - Constrained optimization of protein-ligand complex
        2. Protein-ligand interaction energy calculation
        3. Explainability analysis (if requested)
        4. Energy decomposition analysis (if requested)

    Optimization:
        - No optimization: Use original structures directly
        - With --optimize: constrained optimization of the protein-ligand complex

    Args:
        ligand_file (str): Path to ligand structure file (SDF, MOL2, XYZ, etc.)
        args: Arguments object containing workflow parameters:
            - optimize (bool): Whether to perform structural optimization
            - opt_radius (float): Constraint radius for optimization
            - optimizer, fmax, steps: Optimization parameters
            - exp_lig (bool): Generate ligand explainability heatmap
            - exp_prot (bool): Generate 2D protein-ligand interaction diagram
            - exp_3d (bool): Generate 3D PyMOL visualization script
            - eda (bool): Perform energy decomposition analysis
            - opt_log (bool): Log detailed optimization information
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
                - 'interaction_energy': Interaction energy (eV)
                - 'binding_energy_kcal_mol': Interaction energy in kcal/mol
                - 'ligand_explainability': Analysis results and components
            - error_message: None if successful, error string if failed

    Output Files Generated:
        Optimization outputs:
            - constrained_opt_complexes/{ligand}_complex_constrained_opt.xyz
            - constrained_opt_components/{protein}_{ligand}_constrained_opt.xyz
            - constrained_opt_components/{ligand}_constrained_opt.xyz

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

        >>> # With explainability
        >>> args.exp_lig = True
        >>> args.eda = True
        >>> result, error = process_single_ligand(
        ...     "ligand.sdf", args, calc, "protein.pdb", output_dir, [], logger
        ... )
    """

    ligand_path = Path(ligand_file)
    ligand_name = ligand_path.stem

    working_ligand_path = ligand_file
    working_complex_path = None

    try:
        # Check if optimization is requested
        if args.optimize:
            working_protein_path, working_ligand_path, working_complex_path = constrained_optimization(
                ligand_file, ligand_name, working_protein_path, output_dir, calc, args, optimization_log, logger
            )
        else:
            # No optimization - use original structures
            working_ligand_path = ligand_file
            working_complex_path = None

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

        result = {
            'ligand_name': ligand_name,
            'ligand_file': ligand_file,
            'interaction_energy': interaction_energy,
            'binding_energy_kcal_mol': interaction_energy * 23.06,
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