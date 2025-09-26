#!/usr/bin/env python3
"""
SO3LR-SF Main Deployment Script

This script provides a command-line interface for running protein-ligand
interaction calculations with SO3LR machine learning force fields.

Features:
- Protein trimming around ligands
- Structure optimization with file existence checks
- Ligand set iteration (single file, directory, or multi-SDF)
- Complex building and energy calculations
- Explainability analysis with heatmap generation

Usage:
    python run_so3lr_sf.py protein.pdb ligands.sdf --trim --optimize --explain
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from src import (
    setup_logging,
    protein_ligand_interaction,
    So3lrSfCalculator
)
from src.structure_ops import trim_structure, optimize_structure, extract_ligands
from src.utils import read_structure
from src.explainability import generate_interaction_heatmap, compute_ligand_energy_differences


def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="SO3LR-SF: Protein-Ligand Interaction Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic calculation
  python run_so3lr_sf.py protein.pdb ligand.sdf

  # Trim protein around ligand (5Å radius)
  python run_so3lr_sf.py protein.pdb ligand.sdf --trim --radius 5.0

  # Optimize structures before calculation
  python run_so3lr_sf.py protein.pdb ligand.sdf --optimize

  # Full workflow with explainability
  python run_so3lr_sf.py protein.pdb ligands.sdf --trim --optimize --explain -o results/

  # Process ligand directory
  python run_so3lr_sf.py protein.pdb ligands_dir/ --optimize --explain -o results/
        """
    )

    # Positional arguments
    parser.add_argument(
        "--protein",
        type=str,
        help="Path to protein structure file"
    )
    parser.add_argument(
        "--ligands",
        type=str,
        help="Path to ligand file, directory with ligands, or multi-SDF file"
    )

    # Main workflow options
    parser.add_argument(
        "--trim",
        action="store_true",
        help="Trim protein around ligand(s) before calculation"
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Optimize structures before energy calculation"
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Generate explainability analysis and heatmaps"
    )

    # Trimming parameters
    parser.add_argument(
        "--radius",
        type=float,
        default=10.0,
        help="Radius in Angstroms for protein trimming (default: 10.0)"
    )
    parser.add_argument(
        "--trim-lig",
        type=str,
        help="Specific ligand file to use for protein trimming (if not specified, uses first ligand)"
    )

    # Optimization parameters
    parser.add_argument(
        "--optimizer",
        choices=["FIRE", "LBFGS"],
        default="FIRE",
        help="Optimization algorithm (default: FIRE)"
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=0.05,
        help="Force convergence criterion in eV/Å (default: 0.05)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Maximum optimization steps (default: 100)"
    )
    
    # Model parameters
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to SO3LR model parameters (auto-detected if not specified)"
    )

    # Logging
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser


def check_optimized_file_exists(original_path: Union[str, Path], output_dir: Optional[Path] = None) -> Optional[str]:
    """
    Check if an optimized version of a file already exists.

    Args:
        original_path: Path to original structure file
        output_dir: Directory where optimized files are saved

    Returns:
        Path to existing optimized file, or None if not found
    """
    original_path = Path(original_path)

    # Determine where optimized file would be saved
    if output_dir:
        optimized_path = output_dir / f"{original_path.stem}_optimized.xyz"
    else:
        optimized_path = original_path.parent / f"{original_path.stem}_optimized.xyz"

    if optimized_path.exists():
        return str(optimized_path)
    return None


def optimize_structure_if_needed(
    structure_path: Union[str, Path],
    calculator: Any,
    output_dir: Optional[Path] = None,
    skip_existing: bool = True,
    optimizer: str = "FIRE",
    fmax: float = 0.05,
    steps: int = 100,
    verbose: bool = False
) -> str:
    """
    Optimize structure, skipping if optimized version already exists.

    Args:
        structure_path: Path to structure to optimize
        calculator: SO3LR calculator instance
        output_dir: Output directory for optimized structure
        skip_existing: Skip if optimized file already exists
        optimizer: Optimization algorithm
        fmax: Force convergence criterion
        steps: Maximum optimization steps
        verbose: Enable verbose logging

    Returns:
        Path to optimized structure (existing or newly created)
    """
    logger = logging.getLogger(__name__)

    # Check if optimized file already exists
    if skip_existing:
        existing_optimized = check_optimized_file_exists(structure_path, output_dir)
        if existing_optimized:
            logger.info(f"Optimized file already exists, skipping: {existing_optimized}")
            return existing_optimized

    # Perform optimization
    logger.info(f"Optimizing structure: {structure_path}")
    optimized_path, opt_info = optimize_structure(
        structure_path,
        calculator=calculator,
        optimizer=optimizer,
        fmax=fmax,
        steps=steps,
        output_dir=output_dir
    )

    if opt_info.get('converged', False):
        logger.info(f"Optimization converged in {opt_info.get('steps', 0)} steps")
    else:
        logger.warning(f"Optimization did not converge after {opt_info.get('steps', 0)} steps")

    logger.info(f"Optimized structure saved: {optimized_path}")
    return optimized_path


def get_ligand_files(ligands_input: str, output_dir: Optional[Path] = None) -> List[str]:
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
        # Check if it's a multi-structure file
        if ligands_path.suffix.lower() == '.sdf':
            try:
                # Try to extract multiple ligands
                extract_dir = output_dir / "individual_ligands" if output_dir else Path("individual_ligands")
                ligand_files = extract_ligands(ligands_path, extract_dir)
                logger.info(f"Extracted {len(ligand_files)} ligands from {ligands_path}")
                return ligand_files
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
        supported_extensions = ['.xyz', '.sdf', '.mol', '.mol2', '.pdb']

        for ext in supported_extensions:
            ligand_files.extend([str(f) for f in ligands_path.glob(f"*{ext}")])

        logger.info(f"Found {len(ligand_files)} ligand files in directory: {ligands_path}")
        return sorted(ligand_files)

    else:
        raise ValueError(f"Invalid ligands input: {ligands_input}")


def main():
    """Main deployment logic."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("Starting SO3LR-SF protein-ligand interaction calculation")
    logger.info(f"Protein: {args.protein}")
    logger.info(f"Ligands: {args.ligands}")
    logger.info(f"Workflow: trim={args.trim}, optimize={args.optimize}, explain={args.explain}")

    # Setup output directory
    output_name = "results"
    if args.optimize:
        output_name += f"_steps_{args.steps}_fmax{args.fmax}"
    if args.trim:
        output_name += f"_trim_{args.radius}A"
    output_dir = Path(args.protein).parent / output_name
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

    # Validate input files
    protein_path = Path(args.protein)
    if not protein_path.exists():
        logger.error(f"Protein file not found: {protein_path}")
        sys.exit(1)

    try:
        # Initialize SO3LR calculator
        logger.info("Initializing SO3LR calculator...")
        calc_kwargs = {}
        if args.explain:
            calc_kwargs['output_per_atom_energy_components'] = True

        calc = So3lrSfCalculator(model_path=args.model_path, **calc_kwargs)
        logger.info(f"Calculator initialized with model: {calc.model_path}")

        # Step 1: Handle trimming (if requested)
        working_protein_path = str(protein_path)

        if args.trim:
            logger.info("=== TRIMMING PHASE ===")

            # Get representative ligand for trimming
            if args.trim_lig:
                # Use specified ligand for trimming
                if not Path(args.trim_lig).exists():
                    logger.error(f"Specified trim ligand not found: {args.trim_lig}")
                    sys.exit(1)
                representative_ligand = args.trim_lig
                logger.info(f"Using specified ligand for trimming: {representative_ligand}")
            else:
                # Use first ligand from ligand set
                ligand_files = get_ligand_files(args.ligands, output_dir)
                if not ligand_files:
                    logger.error("No ligand files found")
                    sys.exit(1)
                representative_ligand = ligand_files[0]
                logger.info(f"Using first ligand for trimming: {representative_ligand}")

            # Perform trimming
            trimmed_protein_path = trim_structure(
                protein_path,
                representative_ligand,
                radius=args.radius,
                output_dir=output_dir
            )
            working_protein_path = trimmed_protein_path
            logger.info(f"Protein trimmed to {args.radius}Å radius: {working_protein_path}")

            # If only trimming was requested, exit here
            if not args.optimize and not args.explain:
                logger.info(f"Trimming complete. Trimmed protein saved: {working_protein_path}")
                return

        # Step 2: Optimization phase
        if args.optimize:
            logger.info("=== OPTIMIZATION PHASE ===")

            # Optimize protein first (if requested)
            logger.info("Optimizing protein...")
            working_protein_path = optimize_structure_if_needed(
                working_protein_path,
                calc._calculator,
                output_dir=output_dir,
                optimizer=args.optimizer,
                fmax=args.fmax,
                steps=args.steps,
                verbose=args.verbose
            )

        # Step 3: Get ligand files and process each
        logger.info("=== LIGAND PROCESSING PHASE ===")
        ligand_files = get_ligand_files(args.ligands, output_dir)
        logger.info(f"Processing {len(ligand_files)} ligands...")

        results = []

        for i, ligand_file in enumerate(ligand_files, 1):
            ligand_path = Path(ligand_file)
            ligand_name = ligand_path.stem

            logger.info(f"\n--- Processing ligand {i}/{len(ligand_files)}: {ligand_name} ---")

            try:
                working_ligand_path = ligand_file

                # Optimize ligand if requested
                if args.optimize:
                    logger.info(f"Optimizing ligand: {ligand_name}")
                    working_ligand_path = optimize_structure_if_needed(
                        ligand_file,
                        calc._calculator,
                        output_dir=output_dir,
                        optimizer=args.optimizer,
                        fmax=args.fmax,
                        steps=args.steps,
                        verbose=False  # Don't spam logs for each ligand
                    )

                # Build complex and optimize if requested
                working_complex_path = None
                if args.optimize:
                    # Create complex by concatenating protein + ligand
                    logger.info(f"Building complex: {ligand_name}")
                    protein_atoms = read_structure(working_protein_path)
                    ligand_atoms = read_structure(working_ligand_path)
                    complex_atoms = protein_atoms + ligand_atoms

                    # Save complex for optimization
                    complex_filename = f"{protein_path.stem}_{ligand_name}_complex.xyz"
                    if output_dir:
                        complex_path = output_dir / complex_filename
                    else:
                        complex_path = Path(complex_filename)

                    from src.utils import write_structure
                    write_structure(complex_atoms, complex_path)
                    logger.info(f"Complex structure saved: {complex_path}")

                    # Optimize complex
                    logger.info(f"Optimizing complex: {ligand_name}")
                    optimized_complex_path = optimize_structure_if_needed(
                        complex_path,
                        calc._calculator,
                        output_dir=output_dir,
                        optimizer=args.optimizer,
                        fmax=args.fmax,
                        steps=args.steps,
                        verbose=False
                    )

                    # Store the optimized complex path for energy calculation
                    working_complex_path = optimized_complex_path

                    # Extract optimized protein and ligand parts from optimized complex
                    optimized_complex_atoms = read_structure(optimized_complex_path)
                    n_protein_atoms = len(protein_atoms)
                    n_ligand_atoms = len(ligand_atoms)

                    # Update working paths to use optimized complex parts
                    optimized_protein_atoms = optimized_complex_atoms[:n_protein_atoms]
                    optimized_ligand_atoms = optimized_complex_atoms[n_protein_atoms:n_protein_atoms + n_ligand_atoms]

                    # Save optimized parts
                    opt_protein_filename = f"{protein_path.stem}_{ligand_name}_protein_opt.xyz"
                    opt_ligand_filename = f"{ligand_name}_from_complex_opt.xyz"

                    if output_dir:
                        opt_protein_path = output_dir / opt_protein_filename
                        opt_ligand_path = output_dir / opt_ligand_filename
                    else:
                        opt_protein_path = Path(opt_protein_filename)
                        opt_ligand_path = Path(opt_ligand_filename)

                    write_structure(optimized_protein_atoms, opt_protein_path)
                    write_structure(optimized_ligand_atoms, opt_ligand_path)

                    working_protein_path = str(opt_protein_path)
                    working_ligand_path = str(opt_ligand_path)

                # Step 4: Calculate interaction energy
                logger.info(f"Calculating interaction energy for: {ligand_name}")


                heatmap_output = None
                
                heatmap_output = output_dir / f"{ligand_name}_heatmap.png"

                # Calculate with explainability
                interaction_energy, analysis = protein_ligand_interaction(
                    working_protein_path,
                    working_ligand_path,
                    model_path=args.model_path,
                    complex_path=working_complex_path,
                    explainability=args.explain,
                    heatmap_output=heatmap_output,
                    verbose=False,
                    **calc_kwargs
                )

                result = {
                    'ligand_name': ligand_name,
                    'ligand_file': ligand_file,
                    'working_protein_path': working_protein_path,
                    'working_ligand_path': working_ligand_path,
                    'working_complex_path': working_complex_path,
                    'interaction_energy': interaction_energy,
                    'binding_energy_kcal_mol': interaction_energy * 23.06,
                    'analysis': analysis
                }

                logger.info(f"  Interaction energy: {interaction_energy:.6f} eV "
                            f"({interaction_energy * 23.06:.2f} kcal/mol)")
                if analysis.get('component_totals'):
                    logger.info("  Component contributions:")
                    for comp, total in analysis['component_totals'].items():
                        logger.info(f"    {comp}: {total:.6f} eV")

                # else:
                #     # Simple energy calculation
                #     interaction_energy = protein_ligand_interaction(
                #         working_protein_path,
                #         working_ligand_path,
                #         model_path=args.model_path,
                #         explainability=False,
                #         verbose=False,
                #         **calc_kwargs
                #     )

                #     result = {
                #         'ligand_name': ligand_name,
                #         'ligand_file': ligand_file,
                #         'working_protein_path': working_protein_path,
                #         'working_ligand_path': working_ligand_path,
                #         'interaction_energy': interaction_energy,
                #         'binding_energy_kcal_mol': interaction_energy * 23.06
                #     }

                #     logger.info(f"  Interaction energy: {interaction_energy:.6f} eV "
                #                f"({interaction_energy * 23.06:.2f} kcal/mol)")

                results.append(result)

            except Exception as e:
                logger.error(f"Error processing {ligand_name}: {e}")
                results.append({
                    'ligand_name': ligand_name,
                    'ligand_file': ligand_file,
                    'error': str(e),
                    'interaction_energy': float('nan'),
                    'binding_energy_kcal_mol': float('nan')
                })

        # Step 5: Summary and results
        logger.info("\n=== RESULTS SUMMARY ===")

        # Filter successful results
        successful_results = [r for r in results if 'error' not in r]
        failed_results = [r for r in results if 'error' in r]

        logger.info(f"Successfully processed: {len(successful_results)}/{len(results)} ligands")

        if failed_results:
            logger.warning(f"Failed to process {len(failed_results)} ligands:")
            for result in failed_results:
                logger.warning(f"  {result['ligand_name']}: {result['error']}")

        if successful_results:
            # Sort by interaction energy (most favorable first)
            successful_results.sort(key=lambda x: x['interaction_energy'])

            logger.info("\nTop 5 binding energies:")
            for i, result in enumerate(successful_results[:5], 1):
                logger.info(f"  {i}. {result['ligand_name']}: "
                           f"{result['interaction_energy']:.4f} eV "
                           f"({result['binding_energy_kcal_mol']:.2f} kcal/mol)")

            # Save results summary if output directory specified
            if output_dir:
                import json
                results_file = output_dir / "results_summary.json"

                # Prepare JSON-serializable results
                json_results = []
                for result in results:
                    json_result = result.copy()
                    # Remove analysis dict for JSON serialization (too complex)
                    if 'analysis' in json_result:
                        json_result['analysis'] = "See individual heatmap files"
                    json_results.append(json_result)

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
                        'results': json_results
                    }, f, indent=2)

                logger.info(f"Results summary saved: {results_file}")

        logger.info("SO3LR-SF calculation workflow complete!")

    except KeyboardInterrupt:
        logger.info("Calculation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()