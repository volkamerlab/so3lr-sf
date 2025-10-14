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
    write_structure,
    get_supported_formats,
    validate_structure
)

from .explainability import (
    generate_energy_heatmap,
    generate_ligand_heatmap,
    generate_protein_interaction_heatmap
)

from .explain_utils import (
    compute_energy_differences
)

# Package metadata
__version__ = "1.0.0"
# __author__ = ""
# __description__ = "SO3LR-SF"

# Main interface functions
__all__ = [
    # Main functions
    'protein_ligand_interaction',
    'setup_logging',

    # Core classes
    'So3lrSfCalculator',

    # Structure operations
    'trim_structure',
    'optimize_structure',
    'extract_ligands',

    # Utilities
    'write_structure',
    'get_supported_formats',
    'validate_structure',

    # Explainability
    'compute_energy_differences',
    'generate_energy_heatmap',
    'generate_ligand_heatmap',
    'generate_protein_interaction_heatmap'
]