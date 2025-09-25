"""
Utility functions for SO3LR-SF

This module contains utility functions for model detection, structure reading,
and file handling operations.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
from ase import Atoms
from ase.io import read


def find_so3lr_params() -> Optional[str]:
    """
    Locate SO3LR model parameters directory within the project.

    This function looks for the SO3LR parameters directory in the project structure:
    - Looks for so3lr/so3lr/params relative to current file or project root
    - Uses the specific path: /home/hamza/github/so3lr-sf/so3lr/so3lr/params

    Returns:
        str: Path to SO3LR parameters directory, or None if not found

    Example:
        >>> path = find_so3lr_params()
        >>> if path:
        ...     print(f"Found SO3LR params at: {path}")
        ... else:
        ...     print("SO3LR parameters not found")
    """
    # Get the project root directory (where this package is located)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent  # Go up from src/ to project root

    # Expected SO3LR params path within the project
    so3lr_params_path = project_root / "so3lr" / "so3lr" / "params"

    # Check if the directory exists
    if so3lr_params_path.is_dir():
        return str(so3lr_params_path)

    # Fallback: check the specific absolute path
    fallback_path = Path("/home/hamza/github/so3lr-sf/so3lr/so3lr/params")
    if fallback_path.is_dir():
        return str(fallback_path)

    return None


def read_structure(file_path: Union[str, Path], index: Union[int, str] = 0) -> Union[Atoms, List[Atoms]]:
    """
    Read molecular structure from various file formats with multi-structure support.

    Supports multiple molecular file formats commonly used in computational chemistry:
    PDB, XYZ, SDF, MOL2, and other formats supported by ASE. Can handle files with
    multiple structures (like multi-frame XYZ or multi-molecule SDF files).

    Args:
        file_path: Path to structure file
        index: Structure index to read:
               - int: specific structure (0 for first)
               - ':' or 'all': all structures
               - '-1': last structure

    Returns:
        Atoms or List[Atoms]: Single structure or list of structures

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported or structure is invalid

    Example:
        >>> # Read single structure
        >>> atoms = read_structure("protein.pdb")
        >>>
        >>> # Read all ligands from multi-molecule SDF
        >>> all_ligands = read_structure("ligands.sdf", index=":")
        >>>
        >>> # Read last frame from trajectory
        >>> final_structure = read_structure("trajectory.xyz", index=-1)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Structure file not found: {file_path}")

    try:
        # Handle different index types
        if index == "all" or index == ":":
            read_index = ":"
        else:
            read_index = index

        # Try reading with ASE's automatic format detection
        result = read(str(file_path), index=read_index)

        # Handle single vs multiple structures
        if isinstance(result, list):
            if len(result) == 0:
                raise ValueError(f"Structure file {file_path} contains no atoms")
            return result
        else:
            if len(result) == 0:
                raise ValueError(f"Structure file {file_path} contains no atoms")
            return result

    except Exception as e:
        raise ValueError(f"Could not read structure from {file_path}: {e}")


def write_structure(atoms: Atoms, file_path: Union[str, Path], format: Optional[str] = None) -> str:
    """
    Write molecular structure to file.

    Args:
        atoms: ASE Atoms object to write
        file_path: Output file path
        format: File format (auto-detected from extension if None)

    Returns:
        str: Path to written file

    Example:
        >>> write_structure(atoms, "output.xyz")
        >>> write_structure(atoms, "output.pdb", format="pdb")
    """
    from ase.io import write

    file_path = Path(file_path)

    # Create directory if it doesn't exist
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        write(str(file_path), atoms, format=format)
        return str(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to write structure to {file_path}: {e}")


def get_supported_formats() -> List[str]:
    """
    Get list of supported file formats for structure reading.

    Returns:
        List[str]: List of supported file extensions
    """
    return ['.xyz', '.pdb', '.sdf', '.mol', '.mol2', '.cif', '.traj', '.vasp', '.poscar']


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