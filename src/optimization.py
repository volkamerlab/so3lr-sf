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

    Args:
        atoms: ASE Atoms object to optimize
        optimizer: Name of the optimizer (case-insensitive)

    Returns:
        Configured optimizer instance

    Raises:
        ValueError: If optimizer is unknown or incompatible
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
    Optimize molecular structure and save as XYZ file.

    This function accepts an ASE Atoms object, optimizes it using the provided
    calculator, and saves the optimized structure as an XYZ file. The structure
    is always saved regardless of convergence status.

    Args:
        atoms: ASE Atoms object to optimize
        calc: So3lrSfCalculator instance to use for optimization
        optimizer: Optimization algorithm (BFGS, BFGSLineSearch, FIRE, FIRE2, GPMin, GoodOldQuasiNewton, LBFGS, LBFGSLineSearch, MDMin, ODE12r, QuasiNewton)
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
        logger.info(f"Output path is already a file: {output_path}, skipping optimization.")
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
        logger.info(f"Applying optimization constraints with radius: {opt_radius} Å")
        constraint = create_optimization_constraint(atoms, n_protein_atoms, opt_radius, protein_path)



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

    # Print status
    if converged:
        logger.info(f"Optimization converged in {nsteps} steps")
    else:
        logger.info(f"Optimization did not converge in {nsteps} steps (max: {steps})")

    if optimization_error:
        logger.warning(f"Optimization encountered issues: {optimization_error}")

    logger.info(f"Max displacement: {max_displacement:.6f} Å")
    logger.info(f"Structure saved to: {output_path_str}")

    return output_path_str, opt_info


def free_optimization(ligand_file: str, ligand_name: str, working_protein_path: str,
                           output_dir: Path, calc, args, optimization_log: Optional[List], logger):
    """
    FREE OPTIMIZATION MODE:
    1. Optimize protein alone (only once, skip if already optimized)
    2. Optimize ligand alone
    3. Build complex from original structures and optimize with constraints

    Returns optimized protein path, optimized ligand path, and optimized complex path
    """
    # Step 1: Optimize protein (only once)
    protein_stem = Path(working_protein_path).stem
    opt_protein_path = output_dir / "free_opt_protein" / f"{protein_stem}_free_opt.xyz"

    if opt_protein_path.exists():
        logger.info(f"Using existing optimized protein: {opt_protein_path}")
        optimized_protein_path = str(opt_protein_path)
    else:
        logger.info(f"Optimizing protein structure: {protein_stem}")
        opt_protein_path.parent.mkdir(parents=True, exist_ok=True)
        protein_atoms = load_ase_structure(working_protein_path)[0]
        optimized_protein_path, protein_opt_info = optimize_structure(
            protein_atoms, calc,
            optimizer=args.optimizer, fmax=args.fmax, steps=args.steps,
            output_path=opt_protein_path
        )

        if args.opt_log and optimization_log is not None:
            optimization_log.append({
                'structure_type': 'protein_free',
                'structure_name': protein_stem,
                'optimization_info': protein_opt_info
            })
        logger.info(f"Protein optimization completed: {optimized_protein_path}")

    # Step 2: Optimize ligand alone
    logger.info(f"Optimizing ligand individually: {ligand_name}")
    ligand_atoms = load_ase_structure(ligand_file)[0]
    opt_ligand_path = output_dir / "free_opt_ligands" / f"{ligand_name}_free_opt.xyz"
    opt_ligand_path.parent.mkdir(parents=True, exist_ok=True)

    optimized_ligand_path, ligand_opt_info = optimize_structure(
        ligand_atoms, calc,
        optimizer=args.optimizer, fmax=args.fmax, steps=args.steps,
        output_path=opt_ligand_path
    )

    if args.opt_log and optimization_log is not None:
        optimization_log.append({
            'structure_type': 'ligand_free',
            'structure_name': ligand_name,
            'optimization_info': ligand_opt_info
        })

    # Step 3: Build complex from originals and optimize with constraints
    logger.info(f"Building and optimizing complex with constraints: {ligand_name}")
    original_protein_atoms = load_ase_structure(working_protein_path)[0]  # Original protein
    original_ligand_atoms = load_ase_structure(ligand_file)[0]  # Original ligand
    complex_atoms = original_protein_atoms + original_ligand_atoms

    opt_complex_path = output_dir / "free_opt_complexes" / f"{ligand_name}_complex_free_opt.xyz"
    opt_complex_path.parent.mkdir(parents=True, exist_ok=True)

    optimized_complex_path, complex_opt_info = optimize_structure(
        complex_atoms, calc,
        optimizer=args.optimizer, fmax=args.fmax, steps=args.steps,
        output_path=opt_complex_path,
        opt_radius=args.opt_radius, n_protein_atoms=len(original_protein_atoms),
        protein_path=working_protein_path
    )

    if args.opt_log and optimization_log is not None:
        optimization_log.append({
            'structure_type': 'complex_free',
            'structure_name': f"{protein_stem}_{ligand_name}_complex",
            'optimization_info': complex_opt_info
        })

    logger.info(f"Free optimization completed for {ligand_name}")
    return optimized_protein_path, optimized_ligand_path, optimized_complex_path


def constrained_optimization(ligand_file: str, ligand_name: str, working_protein_path: str,
                                  output_dir: Path, calc, args, optimization_log: Optional[List], logger):
    """
    CONSTRAINED OPTIMIZATION MODE:
    1. Build complex from original structures
    2. Optimize complex with constraints only
    3. Extract optimized components from complex for energy calculations

    Returns paths to extracted optimized protein, ligand, and complex
    """
    logger.info(f"Building and performing constrained optimization of complex: {ligand_name}")

    # Build complex from original structures
    protein_atoms = load_ase_structure(working_protein_path)[0]
    ligand_atoms = load_ase_structure(ligand_file)[0]
    complex_atoms = protein_atoms + ligand_atoms

    # Optimize the complex with constrained optimization
    opt_complex_path = output_dir / "constrained_opt_complexes" / f"{ligand_name}_complex_constrained_opt.xyz"
    opt_complex_path.parent.mkdir(parents=True, exist_ok=True)

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

    opt_protein_path = components_dir / f"{Path(working_protein_path).stem}_{ligand_name}_protein_constrained_opt.xyz"
    opt_ligand_path = components_dir / f"{ligand_name}_ligand_constrained_opt.xyz"

    write_structure(optimized_protein_atoms, opt_protein_path)
    write_structure(optimized_ligand_atoms, opt_ligand_path)

    logger.info(f"Constrained optimization completed. Using extracted components for energy calculation.")
    return str(opt_protein_path), str(opt_ligand_path), optimized_complex_path


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
        # Handle optimization based on mode
        if args.optimize and hasattr(args, 'optimization_mode'):
            if args.optimization_mode == "free":
                # FREE OPTIMIZATION MODE: optimize protein once, ligand individually, then complex
                working_protein_path, working_ligand_path, working_complex_path = free_optimization(
                    ligand_file, ligand_name, working_protein_path, output_dir, calc, args, optimization_log, logger
                )
            elif args.optimization_mode == "constrained":
                # CONSTRAINED OPTIMIZATION MODE: only optimize complex with constraints
                working_protein_path, working_ligand_path, working_complex_path = constrained_optimization(
                    ligand_file, ligand_name, working_protein_path, output_dir, calc, args, optimization_log, logger
                )
            else:
                logger.warning(f"Unknown optimization mode: {args.optimization_mode}. Skipping optimization.")
        elif args.optimize:
            # Backward compatibility: default to free optimization if no mode specified
            logger.info("No optimization_mode specified, defaulting to 'free' optimization")
            working_protein_path, working_ligand_path, working_complex_path = free_optimization(
                ligand_file, ligand_name, working_protein_path, output_dir, calc, args, optimization_log, logger
            )

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