"""
SO3LR-SF: Main interface module

This module provides the main functions for energy calculations, optimization,
and explainability analysis using SO3LR.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Dict, Any, Tuple

from .calculator import So3lrSfCalculator
from .structure_ops import trim_structure, optimize_structure, extract_ligands
from .utils import read_structure, write_structure
from .explainability import compute_ligand_energy_differences, generate_interaction_heatmap


def setup_logging(verbose: bool = False):
    """
    Setup logging configuration.

    Args:
        verbose: If True, set logging level to DEBUG, otherwise INFO
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def energy_calc_fn(
    structure: Union[str, Path],
    model_path: Optional[str] = None,
    verbose: bool = False,
    **kwargs
) -> float:
    """
    Simple function interface for single energy calculations.

    Args:
        structure: Path to structure file or ASE Atoms object
        model_path: Path to SO3LR model parameters (auto-detected if None)
        verbose: Enable verbose logging
        **kwargs: Additional calculator parameters

    Returns:
        float: Potential energy in eV

    Example:
        >>> energy = energy_calc_fn("molecule.xyz", verbose=True)
        >>> print(f"Energy: {energy:.3f} eV")
    """
    if verbose:
        setup_logging(verbose=True)

    logger = logging.getLogger(__name__)

    logger.info(f"Calculating energy for structure: {structure}")
    calc = So3lrSfCalculator(model_path=model_path, **kwargs)
    energy = calc.calculate_energy(structure)
    logger.info(f"Energy calculation complete: {energy:.6f} eV")

    return energy


def protein_ligand_interaction(
    protein_path: Union[str, Path],
    ligand_path: Union[str, Path],
    model_path: Optional[str] = None,
    explainability: bool = False,
    heatmap_output: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    **calc_kwargs
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """
    Calculate protein-ligand interaction energy with optional explainability.

    This is the main function that calculates energies for protein, ligand, and complex
    in a single workflow. The interaction energy is: E_complex - E_protein - E_ligand

    Args:
        protein_path: Path to protein structure file
        ligand_path: Path to ligand structure file
        model_path: Path to SO3LR model parameters (auto-detected if None)
        explainability: If True, calculate per-atom energy differences and generate heatmap
        heatmap_output: Path to save heatmap image (only used if explainability=True)
        verbose: Enable verbose logging
        **calc_kwargs: Additional calculator parameters

    Returns:
        float or tuple:
            - If explainability=False: Just the interaction energy in eV
            - If explainability=True: (interaction_energy, analysis_dict)

    Example:
        >>> # Simple interaction energy
        >>> interaction = protein_ligand_interaction("protein.pdb", "ligand.sdf", verbose=True)
        >>> print(f"Interaction energy: {interaction:.3f} eV")
        >>>
        >>> # With explainability and heatmap
        >>> interaction, analysis = protein_ligand_interaction(
        ...     "protein.pdb", "ligand.sdf",
        ...     explainability=True,
        ...     heatmap_output="interaction_heatmap.png",
        ...     verbose=True
        ... )
        >>> print(f"Binding energy: {analysis['binding_energy_kcal_mol']:.1f} kcal/mol")
    """
    if verbose:
        setup_logging(verbose=True)

    logger = logging.getLogger(__name__)

    logger.info(f"Starting protein-ligand interaction calculation")
    logger.info(f"Protein: {protein_path}")
    logger.info(f"Ligand: {ligand_path}")
    logger.info(f"Explainability: {explainability}")

    # Enable per-atom components if explainability is requested
    if explainability:
        calc_kwargs['output_per_atom_energy_components'] = True
        logger.debug("Per-atom energy components enabled for explainability")

    # Create calculator
    calc = So3lrSfCalculator(model_path=model_path, **calc_kwargs)
    logger.debug(f"Calculator initialized with model: {calc.model_path}")

    # Read structures
    logger.debug("Reading protein structure...")
    protein_atoms = read_structure(protein_path)
    logger.debug(f"Protein loaded: {len(protein_atoms)} atoms")

    logger.debug("Reading ligand structure...")
    ligand_atoms = read_structure(ligand_path)
    logger.debug(f"Ligand loaded: {len(ligand_atoms)} atoms")

    # Create complex by concatenating protein and ligand
    complex_atoms = protein_atoms + ligand_atoms
    logger.debug(f"Complex created: {len(complex_atoms)} total atoms")

    logger.info("Calculating protein energy...")
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

    logger.info("Calculating complex energy...")
    complex_energy = calc.calculate_energy(complex_atoms)
    logger.debug(f"Complex energy: {complex_energy:.6f} eV")
    complex_components = None
    if explainability:
        complex_components = calc.get_per_atom_energy_components()
        logger.debug("Complex per-atom components extracted")

    # Calculate interaction energy: complex - protein - ligand
    interaction_energy = complex_energy - protein_energy - ligand_energy
    logger.info(f"Interaction energy calculated: {interaction_energy:.6f} eV")
    logger.info(f"Binding energy: {interaction_energy * 23.06:.1f} kcal/mol")

    if not explainability:
        return interaction_energy

    # Explainability analysis
    logger.info("Starting explainability analysis...")

    # Get atom counts
    n_protein_atoms = len(protein_atoms)
    n_ligand_atoms = len(ligand_atoms)
    logger.debug(f"Atom counts - Protein: {n_protein_atoms}, Ligand: {n_ligand_atoms}")

    # Compute ligand energy differences
    logger.debug("Computing per-atom energy differences for ligand atoms...")
    ligand_energy_differences = compute_ligand_energy_differences(
        protein_components, ligand_components, complex_components,
        n_protein_atoms, n_ligand_atoms
    )

    if ligand_energy_differences:
        logger.debug("Per-atom energy differences calculated:")
        for comp, values in ligand_energy_differences.items():
            total = np.sum(values)
            logger.debug(f"  {comp}: {total:.6f} eV (sum of {len(values)} atoms)")

    # Generate heatmap if requested
    heatmap_path = None
    if heatmap_output and ligand_energy_differences:
        logger.info(f"Generating heatmap: {heatmap_output}")
        try:
            ligand_name = Path(ligand_path).stem
            title = f"Protein-Ligand Interaction: {ligand_name}"

            fig = generate_interaction_heatmap(
                ligand_path, ligand_energy_differences, heatmap_output, title
            )
            heatmap_path = str(heatmap_output)
            logger.info(f"Heatmap saved successfully: {heatmap_path}")

            # Clean up matplotlib figure
            import matplotlib.pyplot as plt
            plt.close(fig)

        except Exception as e:
            logger.warning(f"Could not generate heatmap: {e}")

    # Prepare analysis results
    analysis = {
        'interaction_energy': interaction_energy,
        'binding_energy_kcal_mol': interaction_energy * 23.06,  # eV to kcal/mol
        'individual_energies': {
            'protein': protein_energy,
            'ligand': ligand_energy,
            'complex': complex_energy
        },
        'ligand_energy_differences': ligand_energy_differences,
        'component_totals': {
            comp: np.sum(values) for comp, values in ligand_energy_differences.items()
        } if ligand_energy_differences else {},
        'atom_counts': {
            'protein': n_protein_atoms,
            'ligand': n_ligand_atoms,
            'complex': len(complex_atoms)
        },
        'heatmap_path': heatmap_path,
        'input_files': {
            'protein': str(protein_path),
            'ligand': str(ligand_path)
        }
    }

    logger.info("Explainability analysis complete")
    return interaction_energy, analysis


def batch_ligand_screening(
    protein_path: Union[str, Path],
    ligands_file_or_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    model_path: Optional[str] = None,
    explainability: bool = False,
    verbose: bool = False,
    **calc_kwargs
) -> List[Dict[str, Any]]:
    """
    Screen multiple ligands against a protein with batch processing.

    Args:
        protein_path: Path to protein structure
        ligands_file_or_dir: Path to multi-structure file (SDF) or directory with ligand files
        output_dir: Directory to save results and heatmaps
        model_path: Path to SO3LR model parameters
        explainability: Whether to generate explainability analysis for each ligand
        verbose: Enable verbose logging
        **calc_kwargs: Additional calculator parameters

    Returns:
        List[Dict]: Results for each ligand with interaction energies and analysis

    Example:
        >>> results = batch_ligand_screening(
        ...     "protein.pdb", "ligands.sdf", "results/",
        ...     explainability=True, verbose=True
        ... )
        >>> # Sort by binding affinity
        >>> sorted_results = sorted(results, key=lambda x: x['interaction_energy'])
        >>> print(f"Best binder: {sorted_results[0]['ligand_name']} "
        ...       f"({sorted_results[0]['interaction_energy']:.3f} eV)")
    """
    if verbose:
        setup_logging(verbose=True)

    logger = logging.getLogger(__name__)

    ligands_path = Path(ligands_file_or_dir)
    logger.info(f"Starting batch ligand screening")
    logger.info(f"Protein: {protein_path}")
    logger.info(f"Ligands source: {ligands_path}")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

    # Handle different input types
    if ligands_path.is_file():
        # Multi-structure file - extract individual ligands
        logger.info(f"Extracting ligands from multi-structure file: {ligands_path}")
        temp_dir = output_dir / "individual_ligands" if output_dir else Path("individual_ligands")
        ligand_files = extract_ligands(ligands_path, temp_dir)
        logger.info(f"Extracted {len(ligand_files)} ligands")
    elif ligands_path.is_dir():
        # Directory with ligand files
        ligand_files = [str(f) for f in ligands_path.glob("*")
                       if f.suffix.lower() in ['.xyz', '.sdf', '.mol', '.mol2', '.pdb']]
        logger.info(f"Found {len(ligand_files)} ligand files in directory")
    else:
        raise ValueError(f"Invalid ligands input: {ligands_path}")

    logger.info(f"Screening {len(ligand_files)} ligands...")

    results = []
    for i, ligand_file in enumerate(ligand_files, 1):
        ligand_path = Path(ligand_file)
        ligand_name = ligand_path.stem

        logger.info(f"Processing ligand {i}/{len(ligand_files)}: {ligand_name}")

        try:
            # Set up heatmap output if explainability is enabled
            heatmap_output = None
            if explainability and output_dir:
                heatmap_output = output_dir / f"{ligand_name}_heatmap.png"

            # Calculate interaction
            if explainability:
                interaction_energy, analysis = protein_ligand_interaction(
                    protein_path, ligand_file, model_path,
                    explainability=True, heatmap_output=heatmap_output,
                    verbose=False, **calc_kwargs  # Don't spam logs for each ligand
                )

                result = {
                    'ligand_name': ligand_name,
                    'ligand_file': str(ligand_file),
                    'interaction_energy': interaction_energy,
                    **analysis
                }
            else:
                interaction_energy = protein_ligand_interaction(
                    protein_path, ligand_file, model_path,
                    explainability=False, verbose=False, **calc_kwargs
                )

                result = {
                    'ligand_name': ligand_name,
                    'ligand_file': str(ligand_file),
                    'interaction_energy': interaction_energy,
                    'binding_energy_kcal_mol': interaction_energy * 23.06
                }

            results.append(result)
            logger.debug(f"  Interaction energy: {interaction_energy:.6f} eV")

        except Exception as e:
            logger.error(f"Error processing {ligand_name}: {e}")
            results.append({
                'ligand_name': ligand_name,
                'ligand_file': str(ligand_file),
                'interaction_energy': np.nan,
                'error': str(e)
            })

    # Sort results by interaction energy (most favorable first)
    valid_results = [r for r in results if not np.isnan(r['interaction_energy'])]
    valid_results.sort(key=lambda x: x['interaction_energy'])

    logger.info(f"Screening complete! Processed {len(valid_results)}/{len(ligand_files)} ligands successfully.")
    if valid_results:
        logger.info(f"Best binder: {valid_results[0]['ligand_name']} "
                   f"({valid_results[0]['interaction_energy']:.3f} eV)")
        logger.info(f"Worst binder: {valid_results[-1]['ligand_name']} "
                   f"({valid_results[-1]['interaction_energy']:.3f} eV)")

    return results


# Convenience functions for common workflows
def trim_and_calculate(
    protein_path: Union[str, Path],
    ligand_path: Union[str, Path],
    radius: float = 10.0,
    model_path: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    **kwargs
) -> Tuple[float, str]:
    """
    Trim protein around ligand and calculate interaction energy.

    Args:
        protein_path: Path to protein structure
        ligand_path: Path to ligand structure
        radius: Trimming radius in Angstroms
        model_path: SO3LR model path
        output_dir: Output directory for trimmed structure
        verbose: Enable verbose logging
        **kwargs: Additional arguments for protein_ligand_interaction

    Returns:
        tuple: (interaction_energy, trimmed_protein_path)
    """
    if verbose:
        setup_logging(verbose=True)

    logger = logging.getLogger(__name__)

    logger.info(f"Trimming protein to {radius}Å around ligand...")
    trimmed_protein_path, _ = trim_structure(protein_path, ligand_path, radius, output_dir)
    logger.info(f"Trimmed protein saved: {trimmed_protein_path}")

    logger.info("Calculating interaction energy with trimmed protein...")
    interaction_energy = protein_ligand_interaction(
        trimmed_protein_path, ligand_path, model_path, verbose=verbose, **kwargs
    )

    return interaction_energy, trimmed_protein_path


def optimize_and_calculate(
    structure_path: Union[str, Path],
    model_path: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    **kwargs
) -> Tuple[float, str]:
    """
    Optimize structure and calculate energy.

    Args:
        structure_path: Path to structure to optimize
        model_path: SO3LR model path
        output_dir: Output directory for optimized structure
        verbose: Enable verbose logging
        **kwargs: Additional arguments for optimize_structure

    Returns:
        tuple: (final_energy, optimized_structure_path)
    """
    if verbose:
        setup_logging(verbose=True)

    logger = logging.getLogger(__name__)

    calc = So3lrSfCalculator(model_path=model_path)

    logger.info(f"Optimizing structure: {structure_path}")
    optimized_path, opt_info = optimize_structure(
        structure_path, calculator=calc._calculator, output_dir=output_dir, **kwargs
    )

    final_energy = opt_info['final_energy']
    logger.info(f"Optimization complete. Final energy: {final_energy:.6f} eV")
    logger.info(f"Optimized structure: {optimized_path}")

    return final_energy, optimized_path