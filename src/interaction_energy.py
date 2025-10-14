"""
Interaction Energy Calculation Module

This module provides functions for calculating protein-ligand interaction energies
and explainability analysis using SO3LR.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Union, Optional, Dict, Any, Tuple
from ase import Atoms
import prolif as plf

from .calculator import So3lrSfCalculator
from .molecule_loader import load_ase_structure
from .explainability import generate_energy_heatmap
from .explain_utils import compute_energy_differences

def prepare_structures(protein_path: Union[str, Path], ligand_path: Union[str, Path],
                      complex_path: Optional[Union[str, Path]] = None, logger=None) -> Tuple[Atoms, Atoms, Atoms]:
    """
    Load and prepare protein, ligand, and complex structures.

    Args:
        protein_path: Path to protein structure file
        ligand_path: Path to ligand structure file
        complex_path: Optional path to pre-built complex structure
        logger: Logger instance

    Returns:
        Tuple of (protein_atoms, ligand_atoms, complex_atoms)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Read structures
    logger.debug("Reading protein structure...")
    protein_atoms = load_ase_structure(protein_path)[0]
    logger.debug(f"Protein loaded: {len(protein_atoms)} atoms")

    logger.debug("Reading ligand structure...")
    ligand_atoms = load_ase_structure(ligand_path)[0]
    logger.debug(f"Ligand loaded: {len(ligand_atoms)} atoms")

    # Use provided complex or create by concatenating protein and ligand
    if complex_path:
        logger.info(f"Using pre-built complex from: {complex_path}")
        complex_atoms = load_ase_structure(complex_path)[0]
        logger.info(f"Complex loaded: {len(complex_atoms)} total atoms")

        # Validate complex structure
        n_protein_atoms = len(protein_atoms)
        n_ligand_atoms = len(ligand_atoms)
        if len(complex_atoms) != n_protein_atoms + n_ligand_atoms:
            logger.warning(f"Complex atom count ({len(complex_atoms)}) != protein ({n_protein_atoms}) + ligand ({n_ligand_atoms})")
    else:
        logger.debug("Creating complex by concatenating protein and ligand")
        # Complex atoms concatenate protein and ligand
        atomic_numbers = np.concatenate((protein_atoms.get_atomic_numbers(), ligand_atoms.get_atomic_numbers()), axis=None)
        positions = np.concatenate((protein_atoms.get_positions(), ligand_atoms.get_positions()), axis=0)
        complex_atoms = Atoms(symbols=atomic_numbers, positions=positions)
        logger.debug(f"Complex created: {len(complex_atoms)} total atoms")

    return protein_atoms, ligand_atoms, complex_atoms


def calculate_individual_energies(protein_atoms: Atoms, ligand_atoms: Atoms, calc: So3lrSfCalculator,
                                explainability: bool = False, logger=None) -> Tuple[float, float, Optional[Dict], Optional[Dict]]:
    """
    Calculate energies for protein and ligand separately.

    Args:
        protein_atoms: Protein structure
        ligand_atoms: Ligand structure
        calc: Calculator instance
        explainability: Whether to extract per-atom components
        logger: Logger instance

    Returns:
        Tuple of (protein_energy, ligand_energy, protein_components, ligand_components)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Calculating protein energy...")
    logger.debug(f"Protein atoms shape: positions={protein_atoms.get_positions().shape}, atomic_numbers={len(protein_atoms.get_atomic_numbers())}")
    protein_energy = calc.calculate_energy(protein_atoms)
    logger.debug(f"Protein energy: {protein_energy:.6f} eV")

    protein_components = None
    if explainability:
        protein_components = calc.get_per_atom_energy_components()
        logger.debug("Protein per-atom components extracted")

    logger.info("Calculating ligand energy...")
    ligand_energy = calc.calculate_energy(ligand_atoms)
    logger.debug(f"Ligand energy: {ligand_energy:.6f} eV")

    ligand_components = None
    if explainability:
        ligand_components = calc.get_per_atom_energy_components()
        logger.debug("Ligand per-atom components extracted")

    return protein_energy, ligand_energy, protein_components, ligand_components


def calculate_complex_energy(complex_atoms: Atoms, calc: So3lrSfCalculator,
                           explainability: bool = False, logger=None) -> Tuple[float, Optional[Dict]]:
    """
    Calculate energy for the protein-ligand complex.

    Args:
        complex_atoms: Complex structure
        calc: Calculator instance
        explainability: Whether to extract per-atom components
        logger: Logger instance

    Returns:
        Tuple of (complex_energy, complex_components)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Calculating complex energy...")
    complex_energy = calc.calculate_energy(complex_atoms)
    logger.debug(f"Complex energy: {complex_energy:.6f} eV")

    complex_components = None
    if explainability:
        complex_components = calc.get_per_atom_energy_components()
        logger.debug("Complex per-atom components extracted")

    return complex_energy, complex_components


def compute_interaction_energy(complex_energy: float, protein_energy: float, ligand_energy: float,
                             logger=None) -> float:
    """
    Calculate interaction energy: complex - protein - ligand.

    Args:
        complex_energy: Energy of the complex
        protein_energy: Energy of the protein
        ligand_energy: Energy of the ligand
        logger: Logger instance

    Returns:
        Interaction energy in eV
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    interaction_energy = complex_energy - protein_energy - ligand_energy
    logger.info(f"Interaction energy calculated: {interaction_energy:.6f} eV")
    logger.info(f"Binding energy: {interaction_energy * 23.06:.1f} kcal/mol")

    return interaction_energy


def compute_eda_analysis(protein_components: Dict, ligand_components: Dict, complex_components: Dict,
                        logger=None) -> Dict[str, Any]:
    """
    Compute Energy Decomposition Analysis - calculate total energy contribution of each component.

    Args:
        protein_components: Per-atom components for protein
        ligand_components: Per-atom components for ligand
        complex_components: Per-atom components for complex
        logger: Logger instance

    Returns:
        Dictionary with total energies for each component across protein, ligand, and complex
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Computing Energy Decomposition Analysis...")

    eda_analysis = {
        'protein_energy_components': {},
        'ligand_energy_components': {},
        'complex_energy_components': {},
        'interaction_energy_components': {}
    }

    # Get all unique component names across all structures
    all_components = set()
    if protein_components:
        all_components.update(protein_components.keys())
    if ligand_components:
        all_components.update(ligand_components.keys())
    if complex_components:
        all_components.update(complex_components.keys())

    # Calculate totals for each component
    for comp_name in all_components:
        # Protein component total
        protein_total = float(np.sum(protein_components.get(comp_name, 0.0))) if protein_components else 0.0
        eda_analysis['protein_energy_components'][comp_name] = protein_total

        # Ligand component total
        ligand_total = float(np.sum(ligand_components.get(comp_name, 0.0))) if ligand_components else 0.0
        eda_analysis['ligand_energy_components'][comp_name] = ligand_total

        # Complex component total
        complex_total = float(np.sum(complex_components.get(comp_name, 0.0))) if complex_components else 0.0
        eda_analysis['complex_energy_components'][comp_name] = complex_total

        # Interaction energy component (complex - protein - ligand)
        interaction_comp_total = complex_total - protein_total - ligand_total
        eda_analysis['interaction_energy_components'][comp_name] = interaction_comp_total

        logger.debug(f"Component '{comp_name}': "
                    f"protein={protein_total:.6f}, ligand={ligand_total:.6f}, "
                    f"complex={complex_total:.6f}, interaction={interaction_comp_total:.6f} eV")

    logger.info("Energy Decomposition Analysis complete")
    return eda_analysis


def analyze_explainability(protein_components: Dict, ligand_components: Dict, complex_components: Dict,
                         protein_atoms: Atoms, ligand_atoms: Atoms, ligand_path: Union[str, Path],
                         heatmap_output: Optional[Union[str, Path]] = None, logger=None,
                         preloaded_protein_prolif: Optional[Tuple[plf.Molecule, Dict[str, List[int]]]] = None) -> Dict[str, Any]:
    """
    Perform explainability analysis and generate heatmap.

    Args:
        protein_components: Per-atom components for protein
        ligand_components: Per-atom components for ligand
        complex_components: Per-atom components for complex
        protein_atoms: Protein structure
        ligand_atoms: Ligand structure
        ligand_path: Path to ligand file
        heatmap_output: Path to save heatmap
        logger: Logger instance
        preloaded_protein_prolif: Optional preloaded protein ProLIF data for enhanced explainability

    Returns:
        Analysis dictionary with energy differences and heatmap path
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Starting explainability analysis...")

    # Get atom counts
    n_protein_atoms = len(protein_atoms)
    n_ligand_atoms = len(ligand_atoms)
    logger.debug(f"Atom counts - Protein: {n_protein_atoms}, Ligand: {n_ligand_atoms}")

    # Compute ligand energy differences
    logger.debug("Computing per-atom energy differences for ligand atoms...")
    if preloaded_protein_prolif is not None:
        
        ligand_energy_differences, protein_energy_differences = compute_energy_differences(
            protein_components, ligand_components, complex_components,
            n_protein_atoms, n_ligand_atoms, protein_mode=True
        )
    else:
        ligand_energy_differences, protein_energy_differences = compute_energy_differences(
            protein_components, ligand_components, complex_components,
            n_protein_atoms, n_ligand_atoms
        )

    # Generate heatmap if requested
    heatmap_path = None
    if heatmap_output:
        logger.info(f"Generating heatmap: {heatmap_output}")
        try:
            ligand_name = Path(ligand_path).stem
            title = f"Protein-Ligand Interaction: {ligand_name}"

            fig = generate_energy_heatmap(
                ligand_path, ligand_energy_differences, heatmap_output, title, protein_energy_differences, preloaded_protein_prolif
            )
            heatmap_path = str(heatmap_output)
            logger.info(f"Heatmap saved successfully: {heatmap_path}")

            # Clean up matplotlib figure
            plt.close(fig)

        except Exception as e:
            logger.warning(f"Could not generate heatmap: {e}")

    # Prepare analysis results
    analysis = {
        'ligand_energy_differences': ligand_energy_differences,
        'component_totals': {
            comp: float(np.sum(values)) for comp, values in ligand_energy_differences.items()
        } if ligand_energy_differences else {},
        'heatmap_path': heatmap_path,
    }

    logger.info("Explainability analysis complete")
    return analysis


def protein_ligand_interaction(
    protein_path: Union[str, Path],
    ligand_path: Union[str, Path],
    calc: So3lrSfCalculator,
    complex_path: Optional[Union[str, Path]] = None,
    explainability: bool = False,
    eda: bool = False,
    heatmap_output: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    preloaded_protein_prolif: Optional[Tuple[plf.Molecule, Dict[str, List[int]]]] = None
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """
    Calculate protein-ligand interaction energy with optional explainability.

    This is the main function that calculates energies for protein, ligand, and complex
    in a single workflow. The interaction energy is: E_complex - E_protein - E_ligand

    Args:
        protein_path: Path to protein structure file
        ligand_path: Path to ligand structure file
        calc: Initialized calculator instance
        complex_path: Optional path to pre-built complex structure
        explainability: If True, calculate per-atom energy differences and generate heatmap
        eda: If True, save individual energy component totals separately
        heatmap_output: Path to save heatmap image (only used if explainability=True)
        verbose: Enable verbose logging
        preloaded_protein_prolif: Optional preloaded protein ProLIF data for enhanced explainability

    Returns:
        float or tuple:
            - If explainability=False and eda=False: Just the interaction energy in eV
            - If explainability=True or eda=True: (interaction_energy, analysis_dict)

    Example:
        >>> calc = So3lrSfCalculator()
        >>> # Simple interaction energy
        >>> interaction = protein_ligand_interaction("protein.pdb", "ligand.sdf", calc, verbose=True)
        >>> print(f"Interaction energy: {interaction:.3f} eV")
        >>>
        >>> # With explainability and heatmap
        >>> interaction, analysis = protein_ligand_interaction(
        ...     "protein.pdb", "ligand.sdf", calc,
        ...     explainability=True,
        ...     heatmap_output="interaction_heatmap.png",
        ...     verbose=True
        ... )
        >>> print(f"Binding energy: {analysis['binding_energy_kcal_mol']:.1f} kcal/mol")
    """
    if verbose:
        from .utils import setup_logging
        setup_logging(verbose=True)

    logger = logging.getLogger(__name__)

    logger.info(f"Protein: {protein_path}")
    logger.info(f"Ligand: {ligand_path}")
    logger.debug(f"Using calculator with model: {calc.model_path}")

    # Check if per-atom components are needed for explainability or EDA
    if (explainability or eda) and not calc.output_per_atom_energy_components:
        logger.warning("Explainability or EDA requested but calculator was not initialized with output_per_atom_energy_components=True")

    # Step 1: Prepare structures
    protein_atoms, ligand_atoms, complex_atoms = prepare_structures(
        protein_path, ligand_path, complex_path, logger
    )

    # Step 2: Calculate individual energies
    protein_energy, ligand_energy, protein_components, ligand_components = calculate_individual_energies(
        protein_atoms, ligand_atoms, calc, explainability or eda, logger
    )

    # Step 3: Calculate complex energy
    complex_energy, complex_components = calculate_complex_energy(
        complex_atoms, calc, explainability or eda, logger
    )

    # Step 4: Compute interaction energy
    interaction_energy = compute_interaction_energy(
        complex_energy, protein_energy, ligand_energy, logger
    )

    if not explainability and not eda:
        return interaction_energy

    # Step 5: Analysis (Explainability and/or EDA)
    analysis = {}

    if explainability:
        # Explainability analysis (enhanced if preloaded ProLIF data is provided)
        explainability_analysis = analyze_explainability(
            protein_components, ligand_components, complex_components,
            protein_atoms, ligand_atoms, ligand_path, heatmap_output, logger,
            preloaded_protein_prolif=preloaded_protein_prolif
        )
        analysis.update(explainability_analysis)

    if eda:
        # Energy Decomposition Analysis
        eda_analysis = compute_eda_analysis(
            protein_components, ligand_components, complex_components, logger
        )
        analysis.update(eda_analysis)

    return interaction_energy, analysis
