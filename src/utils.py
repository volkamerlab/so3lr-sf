"""
Utility functions for SO3LR-SF

This module contains utility functions for model detection, structure reading,
and file handling operations.
"""
import json
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
from ase import Atoms
from ase.io import write

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

def write_structure(atoms: Atoms, file_path: Path) -> str:
    """
    Write molecular structure to file.

    Args:
        atoms: ASE Atoms object to write
        file_path: Output file path (format auto-detected from extension)

    Returns:
        str: Path to written file

    Example:
        >>> write_structure(atoms, "output.xyz")
        >>> write_structure(atoms, "output.pdb")
    """
    logger.debug(f"Writing structure with {len(atoms)} atoms to {file_path}")

    # Create directory if it doesn't exist
    if not file_path.parent.exists():
        logger.debug(f"Creating directory: {file_path.parent}")
        file_path.parent.mkdir(parents=True, exist_ok=True)
    if not file_path.parent.exists():
        logger.debug(f"Creating directory: {file_path.parent}")
        file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        write(file_path, atoms)
        logger.debug(f"Structure successfully written to {file_path}")
        return str(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to write structure to {file_path}: {e}")


def get_supported_formats() -> List[str]:
    """
    Get list of supported file formats for structure reading.

    Returns:
        List[str]: List of supported file extensions
    """
    return ['.xyz', '.pdb', '.sdf']


def validate_structure(atoms: Atoms) -> bool:
    """
    Validate that an ASE Atoms object is suitable for calculations.

    Args:
        atoms: ASE Atoms object to validate

    Returns:
        bool: True if structure is valid

    Raises:
        ValueError: If structure is invalid with description of the problem
    """
    if len(atoms) == 0:
        raise ValueError("Structure contains no atoms")

    if not hasattr(atoms, 'positions') or atoms.positions is None:
        raise ValueError("Structure has no atomic positions")

    if len(atoms.positions) != len(atoms):
        raise ValueError("Number of positions doesn't match number of atoms")

    # Check for NaN or infinite coordinates
    if np.any(np.isnan(atoms.positions)) or np.any(np.isinf(atoms.positions)):
        raise ValueError("Structure contains invalid (NaN or infinite) coordinates")

    return True

def get_ligand_files(ligands_input: str, output_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """
    Get list of ligand files from input (single file, directory, or multi-SDF).

    Args:
        ligands_input: Path to ligands (file or directory)
        output_dir: Output directory for extracted ligands

    Returns:
        List of ligand file paths
    """
    ligands_path = Path(ligands_input)

    if not ligands_path.exists():
        raise FileNotFoundError(f"Ligands input not found: {ligands_input}")

    if ligands_path.is_file():
        # Check if it's a multi-structure file (SDF or XYZ)
        if ligands_path.suffix.lower() in ['.sdf', '.xyz', '.pdb']:
            try:
                # Try to extract multiple ligands
                from .molecule_loader import extract_ligands
                extract_dir = output_dir / "individual_ligands" if output_dir else Path("individual_ligands")
                ligand_files = extract_ligands(ligands_path, extract_dir)

                # Only return extracted files if we found multiple structures
                if len(ligand_files) > 1:
                    logger.info(f"Extracted {len(ligand_files)} ligands from {ligands_path}")
                    return ligand_files
                else:
                    # Single structure - return original file
                    logger.info(f"Single structure found in {ligands_path}")
                    return [str(ligands_path)]
            except:
                # Fallback to treating as single ligand
                logger.info(f"Treating {ligands_path} as single ligand file")
                return [str(ligands_path)]
        else:
            # Single ligand file
            logger.info(f"Using single ligand file: {ligands_path}")
            return [str(ligands_path)]

    elif ligands_path.is_dir():
        # Directory with ligand files
        ligand_files = []
        supported_extensions = ['.xyz', '.sdf', '.pdb']

        for ext in supported_extensions:
            ligand_files.extend([str(f) for f in ligands_path.glob(f"*{ext}")])

        logger.info(f"Found {len(ligand_files)} ligand files in directory: {ligands_path}")
        return sorted(ligand_files)

    else:
        raise ValueError(f"Invalid ligands input: {ligands_input}")


def write_opt_structure(
    atoms: Atoms,
    structure_type: str,
    filename: str,
    output_dir: Optional[Union[str, Path]] = None
) -> str:
    """
    Save structure in organized subdirectories based on structure type.

    Creates subdirectories dynamically based on structure type and saves files there.

    Args:
        atoms: ASE Atoms object to save
        structure_type: Type of structure ('complex', 'opt_ligand', 'protein', etc.)
        filename: Name for the output file (with extension)
        output_dir: Base output directory (default: current directory)

    Returns:
        str: Full path to saved file

    Example:
        >>> # Save complex structure
        >>> path = write_opt_structure(
        ...     complex_atoms, "complex",
        ...     "alanine_water_complex.xyz", "results/"
        ... )
        >>>
        >>> # Save optimized ligand
        >>> path = write_opt_structure(
        ...     ligand_atoms, "opt_ligand",
        ...     "water_optimized.xyz", "results/"
        ... )
    """
    # Set up base directory
    if output_dir is None:
        base_dir = Path(".")
    else:
        base_dir = Path(output_dir)

    # Create subdirectory based on structure type
    type_dir = base_dir / structure_type
    type_dir.mkdir(parents=True, exist_ok=True)

    # Create full path and save
    file_path = type_dir / filename
    return write_structure(atoms, file_path)


def setup_output_directory(protein_path: Union[str, Path], optimize: bool = False, trim_radius: Optional[float] = None,
                          exp_lig: bool = False, exp_prot: bool = False, exp_3d: bool = False,
                          steps: Optional[int] = None, fmax: Optional[float] = None) -> Path:
    """Setup and create output directory based on workflow parameters.

    Args:
        protein_path: Path to protein structure file
        optimize: Whether structure optimization will be performed
        trim_radius: Trimming radius in Angstroms (None if no trimming)
        exp_lig: Generate ligand explainability analysis and heatmaps
        exp_prot: Generate protein explainability with interaction analysis
        exp_3d: Generate 3D PyMOL visualization of protein energy components
        steps: Optimization steps (for directory naming)
        fmax: Force maximum for optimization (for directory naming)

    Returns:
        Path: Created output directory path
    """
    
    output_name = "results"
    if optimize:
        output_name += f"_steps_{steps}_fmax{fmax}"
    if trim_radius is not None:
        output_name += f"_trim_{trim_radius}A"

    output_dir = Path(protein_path).parent / output_name
    logger.debug(f"Creating output directory: {output_dir}")
    logger.debug(f"Creating output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Note: Optimization subdirectories are created dynamically by optimization functions as needed
        
    # Create explain subdirectories based on specific modes
    if exp_lig:
        logger.debug(f"Created optimization subdirectories in: {output_dir}")
        
    # Create explain subdirectories based on specific modes
    if exp_lig:
        (output_dir / "ligand_exp").mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created ligand explainability subdirectory: {output_dir / 'ligand_exp'}")

    if exp_prot:
        (output_dir / "pl_2d_exp").mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created protein explainability subdirectory: {output_dir / 'pl_2d_exp'}")

    if exp_3d:
        (output_dir / "pl_3d_exp").mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created 3D explainability subdirectory: {output_dir / 'pl_3d_exp'}")
        logger.debug(f"Created ligand explainability subdirectory: {output_dir / 'ligand_exp'}")

    if exp_prot:
        (output_dir / "pl_2d_exp").mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created protein explainability subdirectory: {output_dir / 'pl_2d_exp'}")

    if exp_3d:
        (output_dir / "pl_3d_exp").mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created 3D explainability subdirectory: {output_dir / 'pl_3d_exp'}")

    return output_dir


def save_results(results, output_dir, args, protein_path, optimization_log, logger):
    """Save results and optimization logs."""

    # Save results summary
    if output_dir:
        results_file = output_dir / "results_summary.json"

        successful_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]

        with open(results_file, 'w') as f:
            json.dump({
                'workflow_parameters': {
                    'protein': str(protein_path),
                    'ligands_source': args.ligands,
                    'trim': args.trim,
                    'trim_radius': args.radius if args.trim else None,
                    'optimize': args.optimize,
                    'optimization_mode': getattr(args, 'optimization_mode', 'no-strain') if args.optimize else None,
                    'opt_radius': getattr(args, 'opt_radius', None) if args.optimize else None,
                    'ligand_strain_calculation': getattr(args, 'optimization_mode', 'no-strain') in ['strain', 'strain-prot'] if args.optimize else False,
                    'protein_strain_calculation': getattr(args, 'optimization_mode', 'no-strain') == 'strain-prot' if args.optimize else False,
                    'ligand explain 2D': args.exp_lig,
                    'PL interactions explain 2D': args.exp_prot,
                    'PL interactions explain 3D': args.exp_3d,
                    'optimizer': args.optimizer if args.optimize else None,
                    'fmax': args.fmax if args.optimize else None,
                    'steps': args.steps if args.optimize else None
                },
                'summary': {
                    'total_ligands': len(results),
                    'successful': len(successful_results),
                    'failed': len(failed_results)
                },
                'results': results
            }, f, indent=2)

        logger.info(f"Results summary saved: {results_file}")

    # Save optimization log
    if args.opt_log and optimization_log:
        logger.debug("Saving optimization log...")
        logger.debug("Saving optimization log...")
        counter = 1
        opt_log_file = output_dir / "optimization_log.json"
        while opt_log_file.exists():
            opt_log_file = output_dir / f"optimization_log_{counter:03d}.json"
            counter += 1

        opt_log_data = {
            'workflow_parameters': {
                'optimizer': args.optimizer,
                'fmax': args.fmax,
                'max_steps': args.steps,
                'protein': str(protein_path),
                'ligands_source': args.ligands
            },
            'optimizations': optimization_log
        }

        with open(opt_log_file, 'w') as f:
            json.dump(opt_log_data, f, indent=2, default=str)

        logger.info(f"Optimization log saved: {opt_log_file}")

def setup_logging(verbose: bool = False, debug: bool = False):
    """
    Setup logging configuration.

    Args:
        verbose: If True, set logging level to INFO for our modules
        debug: If True, set logging level to DEBUG for our modules (overrides verbose)
        verbose: If True, set logging level to INFO for our modules
        debug: If True, set logging level to DEBUG for our modules (overrides verbose)
    """
    # Clear any existing handlers to avoid conflicts
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Set root logger to INFO to prevent spam from other libraries
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m-%d %H:%M:%S',
        force=True
    )

    # Set our application loggers based on debug/verbose flags
    if debug or verbose:
        our_loggers = [
            logging.getLogger('src'),
            logging.getLogger('__main__'),
            logging.getLogger('run_so3lr_sf')
        ]
        log_level = logging.DEBUG if debug else logging.INFO
        log_level = logging.DEBUG if debug else logging.INFO
        for logger in our_loggers:
            logger.setLevel(log_level)
            logger.setLevel(log_level)

    # Suppress verbose external libraries
    external_loggers = [
        'jax', 'MLFF', 'orbax', 'checkpoint', 'so3lr',
        'jax._src', 'jax._src.cache_key', 'jax._src.compiler',
        'jax._src.xla_bridge', 'absl'
    ]
    for logger_name in external_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)