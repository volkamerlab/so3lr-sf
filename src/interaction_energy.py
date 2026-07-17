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
                      complex_path: Optional[Union[str, Path]] = None, logger=None,
                      charges: Tuple[int, int, int] = (0, 0, 0)) -> Tuple[Atoms, Atoms, Atoms]:
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
    protein_atoms.info['charge'] = charges[0]
    ligand_atoms.info['charge'] = charges[1]
    complex_atoms.info['charge'] = charges[2]
    return protein_atoms, ligand_atoms, complex_atoms


def calculate_individual_energies(protein_atoms: Atoms, ligand_atoms: Atoms, calc: So3lrSfCalculator,
                                explainability: bool = False, logger=None) -> Tuple[float, float, Optional[Dict], Optional[Dict]]:
    """
    Calculate energies for protein and ligand non interacting.

    Args:
        protein_atoms: Protein structure
        ligand_atoms: Ligand structure
        calc: Calculator instance
        explainability: Whether to extract per-atom components
        logger: Logger instance

    Returns:
        Tuple of (non_interaction_energy, protein_components, ligand_components)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Calculating non-interacting energy...")
    atomic_numbers = np.concatenate((protein_atoms.get_atomic_numbers(), ligand_atoms.get_atomic_numbers()), axis=None)
    # TODO: Move ligand far away to minimize interactions in a clever way to ensure it is further than cutoff
    positions = np.concatenate((protein_atoms.get_positions(), ligand_atoms.get_positions()+1000), axis=0) # Move ligand far away
    complex_atoms = Atoms(symbols=atomic_numbers, positions=positions)
    # This combined (but separated) system is evaluated in a single call, so its
    # total charge must be the sum of the protein and ligand charges — otherwise
    # the non-interacting reference would silently be computed as neutral.
    complex_atoms.info['charge'] = protein_atoms.info.get('charge', 0) + ligand_atoms.info.get('charge', 0)
    non_interaction_energy = calc.calculate_energy(complex_atoms)
    protein_atoms = len(protein_atoms.get_atomic_numbers())

    protein_components = None
    ligand_components = None
    if explainability:
        non_iter_energy_components = calc.get_per_atom_energy_components()
        protein_components = {k: v[:protein_atoms] for k, v in non_iter_energy_components.items()}
        logger.debug("Protein per-atom components extracted")
        ligand_components = {k: v[protein_atoms:] for k, v in non_iter_energy_components.items()}
        logger.debug("Ligand per-atom components extracted")

    return non_interaction_energy, protein_components, ligand_components


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


def compute_interaction_energy(complex_energy: float, non_iter_energy: float,
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

    interaction_energy = complex_energy - non_iter_energy
    logger.info(f"Interaction energy calculated: {interaction_energy:.6f} eV")
    logger.info(f"Binding energy: {interaction_energy * 23.06:.1f} kcal/mol")

    return interaction_energy


def calculate_strain_energies(optimized_ligand_path: str, optimized_protein_path: str,
                             free_ligand_path: Optional[str], free_protein_path: Optional[str],
                             calc: So3lrSfCalculator, calculate_protein_strain: bool = False,
                             logger=None) -> Dict[str, float]:
    """
    Calculate strain energies for ligand and optionally protein.

    Strain energy = Energy_in_complex - Energy_free_optimized

    Args:
        optimized_ligand_path: Path to ligand extracted from optimized complex
        optimized_protein_path: Path to protein extracted from optimized complex
        free_ligand_path: Path to free optimized ligand (required for strain calc)
        free_protein_path: Path to free optimized protein (optional, for protein strain)
        calc: Calculator instance
        calculate_protein_strain: Whether to calculate protein strain energy
        logger: Logger instance

    Returns:
        Dictionary containing strain energies
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    strain_energies = {}

    # Calculate ligand strain energy
    if free_ligand_path is not None:
        logger.info("Calculating ligand strain energy...")

        # Energy of ligand in optimized complex
        ligand_in_complex = load_ase_structure(optimized_ligand_path)[0]
        ligand_complex_energy = calc.calculate_energy(ligand_in_complex)

        # Energy of free optimized ligand
        free_ligand = load_ase_structure(free_ligand_path)[0]
        free_ligand_energy = calc.calculate_energy(free_ligand)

        ligand_strain = ligand_complex_energy - free_ligand_energy
        strain_energies['ligand_strain'] = ligand_strain

        logger.info(f"Ligand strain energy: {ligand_strain:.6f} eV ({ligand_strain * 23.06:.2f} kcal/mol)")

    # Calculate protein strain energy (if requested)
    if calculate_protein_strain and free_protein_path is not None:
        logger.info("Calculating protein strain energy...")

        # Energy of protein in optimized complex
        protein_in_complex = load_ase_structure(optimized_protein_path)[0]
        protein_complex_energy = calc.calculate_energy(protein_in_complex)

        # Energy of free optimized protein
        free_protein = load_ase_structure(free_protein_path)[0]
        free_protein_energy = calc.calculate_energy(free_protein)

        protein_strain = protein_complex_energy - free_protein_energy
        strain_energies['protein_strain'] = protein_strain

        logger.info(f"Protein strain energy: {protein_strain:.6f} eV ({protein_strain * 23.06:.2f} kcal/mol)")

    return strain_energies

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
                         logger=None,
                         preloaded_protein_prolif: Optional[Tuple[plf.Molecule, Dict[str, List[int]]]] = None,
                         interaction_energy: Optional[float] = None,
                         exp_outputs: Tuple[Union[str, Path], Union[str, Path], Union[str, Path]] = None,
                         ) -> Dict[str, Any]:
    """
    Perform explainability analysis and generate heatmap.

    Args:
        protein_components: Per-atom components for protein
        ligand_components: Per-atom components for ligand
        complex_components: Per-atom components for complex
        protein_atoms: Protein structure
        ligand_atoms: Ligand structure
        ligand_path: Path to ligand file
        exp_lig_heatmap: Path to save heatmap
        logger: Logger instance
        preloaded_protein_prolif: Optional preloaded protein ProLIF data for enhanced explainability
        interaction_energy: Optional interaction energy value

    Returns:
        Analysis dictionary with energy differences and heatmap path
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Starting explainability analysis...")

    exp_pl_3d_output = exp_outputs[-1]
    # Get atom counts
    n_protein_atoms = len(protein_atoms)
    n_ligand_atoms = len(ligand_atoms)
    logger.debug(f"Atom counts - Protein: {n_protein_atoms}, Ligand: {n_ligand_atoms}")

    # Compute ligand energy differences
    logger.debug("Computing per-atom energy differences for ligand atoms...")
    if preloaded_protein_prolif or exp_pl_3d_output:
        ligand_energy_differences, protein_energy_differences = compute_energy_differences(
            protein_components, ligand_components, complex_components,
            n_protein_atoms, n_ligand_atoms, protein_mode=True
        )
    else:
        ligand_energy_differences, protein_energy_differences = compute_energy_differences(
            protein_components, ligand_components, complex_components,
            n_protein_atoms, n_ligand_atoms
        )

    if any(exp_outputs):
        logger.info(f"Generating explainability heatmap ...")
        try:
            ligand_name = Path(ligand_path).stem
            title = f"Protein-Ligand Interaction: {ligand_name}"
            fig = generate_energy_heatmap(
                ligand_path, ligand_energy_differences, exp_outputs, title, protein_energy_differences, preloaded_protein_prolif, interaction_energy,
                protein_atoms if exp_pl_3d_output else None
            )
            
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
    }
    exp_lig_heatmap = exp_outputs[0] if exp_outputs else None
    exp_pl_2d_output = exp_outputs[1] if exp_outputs else None
    exp_pl_3d_output = exp_outputs[2] if exp_outputs else None
    
    # Add output paths only if they are not None
    if exp_lig_heatmap is not None and exp_lig_heatmap.exists():
        analysis['ligand_explainability_heatmap'] = str(exp_lig_heatmap)
    if exp_pl_2d_output is not None and exp_pl_2d_output.exists():
        analysis['pl_2D_interactions_heatmap'] = str(exp_pl_2d_output)
    if exp_pl_3d_output is not None and exp_pl_3d_output.exists():
        analysis['pl_3D_interactions_session'] = str(exp_pl_3d_output)

    logger.info("Explainability analysis complete")
    return analysis


def protein_ligand_interaction(
    protein_path: Union[str, Path],
    ligand_path: Union[str, Path],
    calc: So3lrSfCalculator,
    complex_path: Optional[Union[str, Path]] = None,
    eda: bool = False,
    verbose: bool = False,
    preloaded_protein_prolif: Optional[Tuple[plf.Molecule, Dict[str, List[int]]]] = None,
    exp_outputs: Optional[Tuple[Optional[Union[str, Path]], Optional[Union[str, Path]], Optional[Union[str, Path]]]] = None,
    charges: Tuple[Optional[int], Optional[int], Optional[int]] = (0, 0, 0),
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
        eda: If True, save individual energy component totals separately
        verbose: Enable verbose logging
        preloaded_protein_prolif: Optional preloaded protein ProLIF data for enhanced explainability
        exp_outputs: Optional tuple of paths for explainability outputs:
            (ligand_heatmap_path, pl_2D_heatmap_path, pl_3D_session_path)

    Returns:
        float or tuple:
            - If None of output path for explainability and eda=False: Just the interaction energy in eV
            - If any of output path for explainability or eda=True: (interaction_energy, analysis_dict)

    Example:
        >>> calc = So3lrSfCalculator()
        >>> # Simple interaction energy
        >>> interaction = protein_ligand_interaction("protein.pdb", "ligand.sdf", calc, verbose=True)
        >>> print(f"Interaction energy: {interaction:.3f} eV")
        >>>
        >>> # With explainability and heatmap
        >>> interaction, analysis = protein_ligand_interaction(
        ...     "protein.pdb", "ligand.sdf", calc,
        ...     exp_outputs=("ligand_heatmap.png", None, None),
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

    exp_mode = any(exp_outputs) if exp_outputs else False

    # Check if per-atom components are needed for explainability or EDA
    if (exp_mode or eda) and not calc.output_per_atom_energy_components:
        logger.warning("Explainability or EDA requested but calculator was not initialized with output_per_atom_energy_components=True")

    # Step 1: Prepare structures
    protein_atoms, ligand_atoms, complex_atoms = prepare_structures(
        protein_path, ligand_path, complex_path, logger, charges=charges
    )

    # Step 2: Calculate individual energies
    non_interaction_energy, protein_components, ligand_components = calculate_individual_energies(
        protein_atoms, ligand_atoms, calc, exp_mode or eda, logger
    )

    # Step 3: Calculate complex energy
    complex_energy, complex_components = calculate_complex_energy(
        complex_atoms, calc, exp_mode or eda, logger
    )

    # Step 4: Compute interaction energy
    interaction_energy = compute_interaction_energy(
        complex_energy, non_interaction_energy, logger
    )

    if not exp_mode and not eda:
        return interaction_energy

    # Step 5: Analysis (Explainability and/or EDA)
    analysis = {}
    if exp_mode or preloaded_protein_prolif is not None:
        # Explainability analysis (enhanced if preloaded ProLIF data is provided)
        explainability_analysis = analyze_explainability(
            protein_components, ligand_components, complex_components,
            protein_atoms, ligand_atoms, ligand_path, logger,
            preloaded_protein_prolif=preloaded_protein_prolif,
            interaction_energy=interaction_energy,
            exp_outputs=exp_outputs
        )
        analysis.update(explainability_analysis)

    if eda:
        # Energy Decomposition Analysis
        eda_analysis = compute_eda_analysis(
            protein_components, ligand_components, complex_components, logger
        )
        analysis.update(eda_analysis)

    return interaction_energy, analysis
