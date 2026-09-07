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
    python run_so3lr_sf.py protein.pdb ligands.sdf --trim --optimize --exp-lig
"""

import argparse
import sys
import logging
from pathlib import Path
from tqdm import tqdm

from src.calculator import So3lrSfCalculator
from src.utils import setup_logging
from src.optimization import process_single_ligand
from src.trim import perform_trimming
from src.utils import setup_output_directory, save_results, get_ligand_files
from src.molecule_loader import load_molecule_to_prolif

def setup_argument_parser() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="SO3LR-SF: Protein-Ligand Interaction Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic calculation
  python so3lr_sf.py protein.pdb ligand.sdf

  # Trim protein around ligand (5Å radius)
  python so3lr_sf.py protein.pdb ligand.sdf --trim 5.0

  # Optimize structures before calculation
  python so3lr_sf.py protein.pdb ligand.sdf --optimize

  # Full workflow with explainability
  python so3lr_sf.py protein.pdb ligands.sdf --trim 10.0 --optimize --exp-lig

  # 3D protein energy visualization
  python so3lr_sf.py protein.pdb ligands.sdf --exp-prot --exp-3d

  # Process ligand directory with double precision
  python so3lr_sf.py protein.pdb ligands.sdf --optimize --exp-lig --dp
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
        "--optimize",
        type=float,
        help="Optimize structures with specified radius in Angstroms before energy calculation"
    )
    parser.add_argument(
        "--exp-lig",
        action="store_true",
        help="Generate explainability analysis and heatmaps"
    )
    parser.add_argument(
        "--exp-prot",
        action="store_true",
        help="Generate enhanced explainability with protein-ligand interaction analysis and residue coloring"
    )
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Perform Energy Decomposition Analysis - save individual energy terms separately"
    )
    parser.add_argument(
        "--exp-3d",
        action="store_true",
        help="Generate 3D PyMOL visualization of protein energy components"
    )

    # Trimming parameters
    parser.add_argument(
        "--trim",
        type=float,
        help="Trim protein around ligand(s) with specified radius in Angstroms"
    )
    parser.add_argument(
        "--trim-lig",
        type=str,
        help="Specific ligand file to use for protein trimming (if not specified, uses first ligand)"
    )

    # Optimization parameters
    parser.add_argument(
        "--optimizer",
        choices=["FIRE", "FIRE2", "LBFGS", "BFGS", "BFGSLineSearch", "LBFGSLineSearch",
                 "GPMin", "MDMin", "ODE12r", "GoodOldQuasiNewton", "QuasiNewton"],
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
        "--dp",
        action="store_true",
        help="Enable double precision (float64) for JAX-MD powered calculations"
    )
    parser.add_argument(
        "--elec-lr-cutoff",
        type=float,
        default=10.0,
        help="Long-range cutoff (Angstroms) for the electrostatic term (default: 10.0). "
             "Recommended: 10 for ranking, 1000 for absolute interaction energies vs DFT."
    )

    # Charge parameters
    parser.add_argument(
        "--charge-lig",
        type=int,
        default=0,
        help="Charge of the ligand (default: 0)"
    )
    parser.add_argument(
        "--charge-prot",
        type=int,
        default=0,
        help="Charge of the protein (default: 0)"
    )
    parser.add_argument(
        "--charge-cpx",
        type=int,
        default=0,
        help="Charge of the complex (default: 0)"
    )

    # Logging
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable logging output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug level logging (includes verbose output)"
    )
    parser.add_argument(
        "--opt-log",
        action="store_true",
        help="Save optimization details to JSON file"
    )

    return parser

def main():
    """Main deployment logic."""
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Setup logging - silent by default, verbose/debug enable logging
    if args.debug or args.verbose:
        setup_logging(verbose=args.verbose, debug=args.debug)
    else:
        # Silent by default
        logging.getLogger().setLevel(logging.CRITICAL + 1)
    logger = logging.getLogger(__name__)

    try:
        # Validate input files
        if not args.protein:
            logger.error("Error: --protein argument is required")
            sys.exit(1)

        if not args.ligands:
            logger.error("Error: --ligands argument is required")
            sys.exit(1)

        protein_path = Path(args.protein)
        if not protein_path.exists():
            logger.error(f"Protein file not found: {protein_path}")
            sys.exit(1)

        ligands_path = Path(args.ligands)
        if not ligands_path.exists():
            logger.error(f"Ligands file or directory not found: {ligands_path}")
            sys.exit(1)

        logger.info("Starting SO3LR-SF protein-ligand interaction calculation")
        logger.debug(f"Command line arguments: {vars(args)}")
        logger.info(f"Protein: {args.protein}")
        logger.info(f"Ligands: {args.ligands}")

        # Warn if charges are not explicitly provided
        if args.charge_lig == 0 and args.charge_prot == 0 and args.charge_cpx == 0:
            logger.warning("WARNING: Charge parameters not explicitly provided. Using default values (charge-lig=0, charge-prot=0, charge-cpx=0). "
                          "For accurate calculations, consider specifying actual charges using --charge-lig, --charge-prot, and --charge-cpx arguments.")
        else:
            logger.info(f"Charges: protein={args.charge_prot}, ligand={args.charge_lig}, complex={args.charge_cpx}")

        # Check for 3D explain argument (handle hyphen conversion)
        logger.info(f"Workflow: trim={args.trim}, optimize={args.optimize}, eda={args.eda}, exp-lig={args.exp_lig}, exp-prot={args.exp_prot}, exp-3d={args.exp_3d}")

        # Setup output directory and subdirectories
        output_dir = setup_output_directory(
            protein_path, optimize=args.optimize, trim_radius=args.trim,
            exp_lig=args.exp_lig, exp_prot=args.exp_prot, exp_3d=args.exp_3d,
            steps=args.steps, fmax=args.fmax
        )
        logger.info(f"Output directory: {output_dir}")
        logger.debug(f"Created output subdirectories for workflow modes")

        # Setup calculator
        calc_kwargs = {}
        if args.exp_lig or args.eda or args.exp_prot or args.exp_3d:
            calc_kwargs['output_per_atom_energy_components'] = True
        if args.dp:
            calc_kwargs['dp'] = True
        calc_kwargs['elec_lr_cutoff'] = args.elec_lr_cutoff

        logger.debug(f"Calculator kwargs: {calc_kwargs}")
        logger.info("Initializing SO3LRSF calculator...")
        calc = So3lrSfCalculator(**calc_kwargs)

        # Initialize optimization log
        optimization_log = [] if args.opt_log else None

        # Step 1: Trimming phase
        working_protein_path = str(protein_path)
        if args.trim is not None:
            working_protein_path = perform_trimming(
                protein_path, args.ligands, args.trim, args.trim_lig, output_dir, logger
            )

        # Store optimization parameters in args for process_single_ligand
        if args.optimize is not None:
            args.opt_radius = args.optimize  # Use optimize value as radius
            logger.info(f"=== OPTIMIZATION (radius: {args.opt_radius}Å) ===")
        else:
            args.opt_radius = None

        # Optional: load ProLIF protein structure for explainability
        preloaded_protein_prolif = None
        if args.exp_prot:
            preloaded_protein_prolif = load_molecule_to_prolif(working_protein_path, is_protein=True)
        # Step 3: Ligand processing
        logger.info("=== LIGAND PROCESSING PHASE ===")
        ligand_files = get_ligand_files(args.ligands, output_dir)
        logger.info(f"Processing {len(ligand_files)} ligands...")

        results = []
        failed_count = 0
        
        for i, ligand_file in enumerate(tqdm(ligand_files, desc="Processing ligands"), 1):
            ligand_name = Path(ligand_file).stem
            logger.info(f"\n--- Processing ligand {i}/{len(ligand_files)}: {ligand_name} ---")

            result, error = process_single_ligand(
                ligand_file, args, calc, working_protein_path,
                output_dir, optimization_log, logger,
                preloaded_protein_prolif,
                charges=(args.charge_prot, args.charge_lig, args.charge_cpx)
            )

            if error:
                logger.error(f"Error processing {ligand_name}: {error}")
                failed_count += 1

            results.append(result)

        # Step 4: Summary and save results
        logger.info("\n=== RESULTS SUMMARY ===")
        successful_results = [r for r in results if 'error' not in r]

        logger.info(f"Successfully processed: {len(successful_results)}/{len(results)} ligands")

        if failed_count > 0:
            logger.warning(f"Failed to process {failed_count} ligands")

        if successful_results:
            # Sort by interaction energy (most favorable first)
            successful_results.sort(key=lambda x: x['interaction_energy'])

            logger.info("\nTop 5 binding energies:")
            for i, result in enumerate(successful_results[:5], 1):
                logger.info(f"  {i}. {result['ligand_name']}: "
                           f"{result['interaction_energy']:.4f} eV "
                           f"({result['binding_energy_kcal_mol']:.2f} kcal/mol)")

        # Save all results
        save_results(results, output_dir, args, protein_path, optimization_log, logger)
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