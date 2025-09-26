"""
SO3LR-SF: Main interface module

This module provides the main functions for energy calculations, optimization,
and explainability analysis using SO3LR.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Dict, Any, Tuple
from ase import Atoms
from tqdm import tqdm

from .calculator import So3lrSfCalculator
from .structure_ops import trim_structure, optimize_structure, extract_ligands
from .utils import read_structure, write_structure
from .explainability import compute_ligand_energy_differences, generate_interaction_heatmap


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


def energy_calc_fn(
    structure: Union[str, Path],
    calc: So3lrSfCalculator,
    verbose: bool = False
) -> float:
    """
    Simple function interface for single energy calculations.

    Args:
        structure: Path to structure file or ASE Atoms object
        calc: Initialized calculator instance
        verbose: Enable verbose logging

    Returns:
        float: Potential energy in eV

    Example:
        >>> calc = So3lrSfCalculator()
        >>> energy = energy_calc_fn("molecule.xyz", calc, verbose=True)
        >>> print(f"Energy: {energy:.3f} eV")
    """
    if verbose:
        setup_logging(verbose=True)

    logger = logging.getLogger(__name__)

    logger.info(f"Calculating energy for structure: {structure}")
    energy = calc.calculate_energy(structure)
    logger.info(f"Energy calculation complete: {energy:.6f} eV")

    return energy


def protein_ligand_interaction(
    protein_path: Union[str, Path],
    ligand_path: Union[str, Path],
    calc: So3lrSfCalculator,
    complex_path: Optional[Union[str, Path]] = None,
    explainability: bool = False,
    heatmap_output: Optional[Union[str, Path]] = None,
    verbose: bool = False
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
        heatmap_output: Path to save heatmap image (only used if explainability=True)
        verbose: Enable verbose logging

    Returns:
        float or tuple:
            - If explainability=False: Just the interaction energy in eV
            - If explainability=True: (interaction_energy, analysis_dict)

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
        setup_logging(verbose=True)

    logger = logging.getLogger(__name__)

    logger.info(f"Protein: {protein_path}")
    logger.info(f"Ligand: {ligand_path}")
    logger.debug(f"Using calculator with model: {calc.model_path}")

    # Check if per-atom components are needed for explainability
    if explainability and not calc.output_per_atom_energy_components:
        logger.warning("Explainability requested but calculator was not initialized with output_per_atom_energy_components=True")

    # Read structures
    logger.debug("Reading protein structure...")
    protein_atoms = read_structure(protein_path)
    logger.debug(f"Protein loaded: {len(protein_atoms)} atoms")

    logger.debug("Reading ligand structure...")
    ligand_atoms = read_structure(ligand_path)
    logger.debug(f"Ligand loaded: {len(ligand_atoms)} atoms")

    # Use provided complex or create by concatenating protein and ligand
    if complex_path:
        logger.info(f"Using pre-built complex from: {complex_path}")
        complex_atoms = read_structure(complex_path)
        logger.info(f"Complex loaded: {len(complex_atoms)} total atoms")

        # When using pre-built complex, extract protein and ligand parts from it
        # to ensure consistent atom ordering with the complex
        n_protein_atoms = len(protein_atoms)
        n_ligand_atoms = len(ligand_atoms)

        if len(complex_atoms) != n_protein_atoms + n_ligand_atoms:
            logger.warning(f"Complex atom count ({len(complex_atoms)}) != protein ({n_protein_atoms}) + ligand ({n_ligand_atoms})")
    else:
        logger.debug("Creating complex by concatenating protein and ligand")
        # complex_atoms concatenate protein and ligand
        atomic_numbers = np.concatenate((protein_atoms.get_atomic_numbers(), ligand_atoms.get_atomic_numbers()),  axis=None)
        positions = np.concatenate((protein_atoms.get_positions(), ligand_atoms.get_positions()),  axis=0)
        complex_atoms = Atoms(symbols=atomic_numbers, positions=positions)
        logger.debug(f"Complex created: {len(complex_atoms)} total atoms")
        

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

    # if ligand_energy_differences:
    #     logger.debug("Per-atom energy differences calculated:")
    #     for comp, values in ligand_energy_differences.items():
    #         total = np.sum(values)
    #         logger.debug(f"  {comp}: {total:.6f} eV (sum of {len(values)} atoms)")

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
        'ligand_energy_differences': ligand_energy_differences,
        'component_totals': {
            comp: float(np.sum(values)) for comp, values in ligand_energy_differences.items()
        } if ligand_energy_differences else {},
        'heatmap_path': heatmap_path,
    }

    logger.info("Explainability analysis complete")
    return interaction_energy, analysis


def batch_ligand_screening(
    protein_path: Union[str, Path],
    ligands_file_or_dir: Union[str, Path],
    calc: So3lrSfCalculator,
    output_dir: Optional[Union[str, Path]] = None,
    explainability: bool = False,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Screen multiple ligands against a protein with batch processing.

    Args:
        protein_path: Path to protein structure
        ligands_file_or_dir: Path to multi-structure file (SDF) or directory with ligand files
        calc: Initialized calculator instance
        output_dir: Directory to save results and heatmaps
        explainability: Whether to generate explainability analysis for each ligand
        verbose: Enable verbose logging

    Returns:
        List[Dict]: Results for each ligand with interaction energies and analysis

    Example:
        >>> calc = So3lrSfCalculator()
        >>> results = batch_ligand_screening(
        ...     "protein.pdb", "ligands.sdf", calc, "results/",
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
    for i, ligand_file in enumerate(tqdm(ligand_files, desc="Screening ligands"), 1):
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
                    protein_path, ligand_file, calc,
                    explainability=explainability, heatmap_output=heatmap_output,
                    verbose=False  # Don't spam logs for each ligand
                )

                result = {
                    'ligand_name': ligand_name,
                    'ligand_file': str(ligand_file),
                    'interaction_energy': interaction_energy,
                    **analysis
                }
            else:
                interaction_energy = protein_ligand_interaction(
                    protein_path, ligand_file, calc,
                    explainability=False, verbose=False
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
    calc: So3lrSfCalculator,
    radius: float = 10.0,
    output_dir: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    **kwargs
) -> Tuple[float, str]:
    """
    Trim protein around ligand and calculate interaction energy.

    Args:
        protein_path: Path to protein structure
        ligand_path: Path to ligand structure
        calc: Initialized calculator instance
        radius: Trimming radius in Angstroms
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
    trimmed_protein_path = trim_structure(protein_path, ligand_path, radius, output_dir)
    logger.info(f"Trimmed protein saved: {trimmed_protein_path}")

    logger.info("Calculating interaction energy with trimmed protein...")
    interaction_energy = protein_ligand_interaction(
        trimmed_protein_path, ligand_path, calc, verbose=verbose, **kwargs
    )

    return interaction_energy, trimmed_protein_path


def optimize_and_calculate(
    structure_path: Union[str, Path],
    calc: So3lrSfCalculator,
    output_dir: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    **kwargs
) -> Tuple[float, str]:
    """
    Optimize structure and calculate energy.

    Args:
        structure_path: Path to structure to optimize
        calc: Initialized calculator instance
        output_dir: Output directory for optimized structure
        verbose: Enable verbose logging
        **kwargs: Additional arguments for optimize_structure

    Returns:
        tuple: (final_energy, optimized_structure_path)
    """
    if verbose:
        setup_logging(verbose=True)

    logger = logging.getLogger(__name__)

    logger.info(f"Optimizing structure: {structure_path}")
    optimized_path, opt_info = optimize_structure(
        structure_path, calculator=calc._calculator, output_dir=output_dir, **kwargs
    )

    final_energy = opt_info['final_energy']
    logger.info(f"Optimization complete. Final energy: {final_energy:.6f} eV")
    logger.info(f"Optimized structure: {optimized_path}")

    return final_energy, optimized_path