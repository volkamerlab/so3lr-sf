"""
Utility functions for SO3LR-SF

This module contains utility functions for model detection, structure reading,
and file handling operations.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
from ase import Atoms
from ase.io import read, write
from rdkit import Chem
import MDAnalysis as mda
import prolif as plf
from .molecule_loader import load_ase_structure


def write_structure(atoms: Atoms, file_path: Union[str, Path]) -> str:
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
    file_path = Path(file_path)

    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        write(str(file_path), atoms)
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


# def rdkit_to_ase(mol):
#     """
#     Convert RDKit molecule to ASE Atoms object.

#     Args:
#         mol: RDKit molecule object

#     Returns:
#         Atoms: ASE Atoms object
#     """
#     conf = mol.GetConformer()
#     symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
#     positions = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
#     positions = [[p.x, p.y, p.z] for p in positions]
#     return Atoms(symbols=symbols, positions=positions)


def load_molecule_to_prolif(file_path: Union[str, Path]):
    """
    Universal function to load any molecule (protein or ligand) to ProLIF (RDKit) format.

    Supports: PDB, SDF, XYZ formats for both proteins and ligands

    Args:
        file_path: Path to structure file

    Returns:
        rdkit.Chem.Mol: RDKit molecule object optimized for ProLIF

    Raises:
        ValueError: If file format is not supported or molecule cannot be loaded
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix not in ['.pdb', '.sdf', '.xyz']:
        raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .pdb, .sdf, .xyz")

    # Load RDKit molecule based on format
    if suffix == '.pdb':
        mol = Chem.MolFromPDBFile(str(file_path), removeHs=False)
    elif suffix == '.sdf':
        mol = Chem.MolFromMolFile(str(file_path))
    elif suffix == '.xyz':
        u = mda.Universe(str(file_path))
        # add "elements" category
        elements = mda.topology.guessers.guess_types(u.atoms.names)
        u.add_TopologyAttr("elements", elements)
        mol = plf.Molecule.from_mda(u)
    if mol is None:
        raise ValueError(f"Could not load molecule from {file_path}")

    return mol


# def load_molecule_to_rdkit(file_path: Union[str, Path]):
#     """
#     Universal function to load any molecule (protein or ligand) to RDKit format.

#     Supports: PDB, SDF, XYZ formats

#     Args:
#         file_path: Path to structure file

#     Returns:
#         rdkit.Chem.Mol: RDKit molecule object

#     Raises:
#         ValueError: If file format is not supported or molecule cannot be loaded
#     """
#     file_path = Path(file_path)
#     suffix = file_path.suffix.lower()

#     # Load RDKit molecule based on format
#     if suffix == '.pdb':
#         mol = Chem.MolFromPDBFile(str(file_path), removeHs=False)
#     elif suffix == '.sdf':
#         mol = Chem.MolFromMolFile(str(file_path))
#     elif suffix == '.xyz':
#         mol = read_xyz_with_bonds(file_path)
#     else:
#         raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .pdb, .sdf, .xyz")

#     if mol is None:
#         raise ValueError(f"Could not load molecule from {file_path}")

#     return mol


def get_ligand_files(ligands_input: str, output_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """
    Get list of ligand files from input (single file, directory, or multi-SDF).

    Args:
        ligands_input: Path to ligands (file or directory)
        output_dir: Output directory for extracted ligands

    Returns:
        List of ligand file paths
    """
    logger = logging.getLogger(__name__)
    ligands_path = Path(ligands_input)

    if not ligands_path.exists():
        raise FileNotFoundError(f"Ligands input not found: {ligands_input}")

    if ligands_path.is_file():
        # Check if it's a multi-structure file (SDF or XYZ)
        if ligands_path.suffix.lower() in ['.sdf', '.xyz', '.pdb']:
            try:
                # Try to extract multiple ligands
                from .structure_ops import extract_ligands
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


def setup_output_directory(protein_path, optimize=False, trim=False, explain=False,
                          steps=None, fmax=None, radius=None):
    """Setup and create output directory based on workflow parameters."""
    output_name = "results"
    if optimize:
        output_name += f"_steps_{steps}_fmax{fmax}"
    if trim:
        output_name += f"_trim_{radius}A"

    output_dir = Path(protein_path).parent / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    if optimize:
        (output_dir / "opt_ligand").mkdir(parents=True, exist_ok=True)
        (output_dir / "opt_complexes").mkdir(parents=True, exist_ok=True)
    if explain:
        (output_dir / "ligand_exp").mkdir(parents=True, exist_ok=True)

    return output_dir


def save_results(results, output_dir, args, protein_path, optimization_log, logger):
    """Save results and optimization logs."""
    import json

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
                    'explain': args.explain,
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


def setup_logging(verbose: bool = False):
    """
    Setup logging configuration.

    Args:
        verbose: If True, set logging level to DEBUG for our modules, otherwise INFO
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

    # Set our application loggers to DEBUG if verbose is requested
    if verbose:
        our_loggers = [
            logging.getLogger('src'),
            logging.getLogger('__main__'),
            logging.getLogger('run_so3lr_sf')
        ]
        for logger in our_loggers:
            logger.setLevel(logging.DEBUG)

    # Suppress verbose external libraries
    external_loggers = [
        'jax', 'MLFF', 'orbax', 'checkpoint', 'so3lr',
        'jax._src', 'jax._src.cache_key', 'jax._src.compiler',
        'jax._src.xla_bridge', 'absl'
    ]
    for logger_name in external_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)