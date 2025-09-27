"""
SO3LR-SF: SO3LR Forces Calculator

A modular Python package for energy calculations, structure optimization,
and explainability analysis using SO3LR machine learning force fields.

Main Functions:
    - energy_calc_fn: Simple energy calculation
    - protein_ligand_interaction: Protein-ligand binding energy with explainability
    - batch_ligand_screening: Screen multiple ligands
    - trim_and_calculate: Trim protein and calculate binding energy
    - optimize_and_calculate: Optimize structure and calculate energy

Classes:
    - So3lrSfCalculator: Core energy calculator
    - ProteinLigandExplainer: Advanced explainability analysis (if needed)

Example:
    >>> from so3lr_sf import protein_ligand_interaction
    >>> interaction, analysis = protein_ligand_interaction(
    ...     "protein.pdb", "ligand.sdf",
    ...     explainability=True, verbose=True
    ... )
    >>> print(f"Binding energy: {analysis['binding_energy_kcal_mol']:.1f} kcal/mol")
"""

# Import main functions and classes
from .interaction_energy import (
    # energy_calc_fn,
    protein_ligand_interaction,
)

from .utils import setup_logging

from .calculator import So3lrSfCalculator

from .structure_ops import (
    trim_structure,
    optimize_structure,
    extract_ligands
)

from .utils import (
    find_so3lr_params,
    read_structure,
    write_structure,
    get_supported_formats,
    validate_structure
)

from .explainability import (
    compute_ligand_energy_differences,
    generate_interaction_heatmap
)

# Package metadata
__version__ = "1.0.0"
# __author__ = ""
# __description__ = "SO3LR-SF"

# Main interface functions
__all__ = [
    # Main functions
    'energy_calc_fn',
    'protein_ligand_interaction',
    'batch_ligand_screening',
    'setup_logging',

    # Core classes
    'So3lrSfCalculator',

    # Structure operations
    'trim_structure',
    'optimize_structure',
    'extract_ligands',

    # Utilities
    'find_so3lr_params',
    'read_structure',
    'write_structure',
    'get_supported_formats',
    'validate_structure',

    # Explainability
    'compute_ligand_energy_differences',
    'generate_interaction_heatmap'
]