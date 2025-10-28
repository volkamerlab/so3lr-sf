"""
3D Protein-Ligand Energy Visualizer for PyMOL

This module creates PyMOL sessions with protein-ligand complexes showing energy-based interactions.
Features:
- Protein backbone shown as cartoon (gray)
- Ligand shown as sticks with energy-based coloring
- Residues with significant energy contributions shown as sticks
- Gray-to-red/blue gradient coloring based on energy magnitude
- Automatic detection of protein-ligand interactions (H-bonds, hydrophobic, ionic, aromatic)
- Only atoms within 10 Å of ligand are analyzed for optimal performance
- Energy threshold filtering to focus on significant contributions (>5% of max energy)

Energy Components Supported: MLFF, ZBL, Electrostatics, Dispersion, Total
Color Scheme: Gray (neutral) → Red (unfavorable) / Blue (favorable)
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Union
from src.molecule_loader import load_ase_structure


logger = logging.getLogger(__name__)





def create_pymol_session(protein_path: Union[str, Path],
                        protein_energy_differences: Dict[str, List[float]],
                        output_dir: Union[str, Path],
                        session_name: str = "protein_energy_visualization") -> str:
    """
    Create a PyMOL session file for protein-ligand complex energy visualization.

    This function creates an optimized PyMOL visualization with:
    - Protein backbone shown as cartoon (gray)
    - Ligand shown as sticks with energy-based gradient coloring
    - Only residues with significant energy contributions (>5% threshold) shown as sticks
    - Gray-to-red gradient for positive (unfavorable) energies
    - Gray-to-blue gradient for negative (favorable) energies
    - Automatic detection of protein-ligand interactions (H-bonds, hydrophobic, ionic, aromatic)
    - Analysis limited to atoms within 10 Å of ligand for performance

    Args:
        protein_path: Path to protein-ligand complex structure file (PDB, XYZ, etc.)
        protein_energy_differences: Dict with energy component names as keys
                                   and per-atom energy differences as values
                                   Components: MLFF, ZBL, Electrostatics, Dispersion, Total
        output_dir: Directory to save PyMOL session and script files
        session_name: Base name for output files (default: "protein_energy_visualization")

    Returns:
        Path to generated PyMOL script file (.pml)
        Also creates: PyMOL session file (.pse), protein PDB file

    Raises:
        ValueError: If protein structure is invalid or energy data is inconsistent
        FileNotFoundError: If protein file doesn't exist

    Example:
        >>> energy_diffs = {
        ...     'MLFF': [-0.1, 0.2, -0.05, ...],      # Per-atom energy differences
        ...     'Electrostatics': [0.15, -0.3, 0.1, ...],
        ...     'Total': [0.05, -0.1, 0.05, ...]
        ... }
        >>> script_path = create_pymol_session(
        ...     'complex.pdb', energy_diffs, 'output_dir', 'my_complex'
        ... )
        >>> print(f"Run: pymol {script_path}")
    """
    protein_path = Path(protein_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate input files
    if not protein_path.exists():
        raise FileNotFoundError(f"Protein file not found: {protein_path}")

    # Load protein structure to get number of atoms
    try:
        protein_atoms = load_ase_structure(protein_path)[0]
        n_atoms = len(protein_atoms)
    except Exception as e:
        raise ValueError(f"Failed to load protein structure: {e}")

    # Validate energy data
    if not protein_energy_differences:
        raise ValueError("No energy differences provided")

    # Expected energy components (in display order)
    expected_components = ['MLFF', 'ZBL', 'Electrostatics', 'Dispersion', 'Total']
    available_components = [comp for comp in expected_components
                           if comp in protein_energy_differences]

    if not available_components:
        raise ValueError(f"No recognized energy components found. "
                        f"Expected: {expected_components}")

    # Validate that all components have correct number of values
    for comp in available_components:
        values = protein_energy_differences[comp]
        if len(values) != n_atoms:
            raise ValueError(f"Energy component '{comp}' has {len(values)} values "
                           f"but protein has {n_atoms} atoms")

    logger.info(f"Creating PyMOL visualization for {n_atoms} atoms with "
                f"{len(available_components)} energy components")

    # Convert protein to PDB format if needed for PyMOL
    protein_pdb_path = output_dir / f"{session_name}_protein.pdb"
    if protein_path.suffix.lower() != '.pdb':
        # Convert to PDB format
        try:
            from ase.io import write
            write(str(protein_pdb_path), protein_atoms, format='pdb')
            logger.info(f"Converted protein to PDB format: {protein_pdb_path}")
        except Exception as e:
            logger.warning(f"Failed to convert to PDB, using original format: {e}")
            protein_pdb_path = protein_path
    else:
        # Copy PDB file to output directory
        import shutil
        shutil.copy2(protein_path, protein_pdb_path)

    # Generate PyMOL script
    script_path = output_dir / f"{session_name}.pml"

    with open(script_path, 'w') as f:
        # Header
        f.write(f"# PyMOL Protein Energy Visualization\n")
        f.write(f"# Generated from: {protein_path}\n")
        f.write(f"# Components: {', '.join(available_components)}\n\n")

        # Load and setup
        f.write("# Clear workspace\n")
        f.write("delete all\n")
        f.write("bg_color white\n\n")

        # Load protein structure for each component
        for i, component in enumerate(available_components):
            structure_name = f"protein_{component.lower()}"

            f.write(f"# Load structure for {component} component\n")
            f.write(f"load {protein_pdb_path.name}, {structure_name}\n")

            # Position structures side by side
            x_offset = i * 50  # 50 Å separation
            f.write(f"translate [{x_offset}, 0, 0], {structure_name}\n")

            # Set basic representation for protein-ligand complex
            f.write(f"# Hide everything first\n")
            f.write(f"hide everything, {structure_name}\n")

            # Show cartoon for protein (assuming protein is the first part of the complex)
            f.write(f"show cartoon, {structure_name} and polymer\n")
            f.write(f"color gray80, {structure_name} and polymer\n")

            # Show sticks for ligand (assuming ligand is organic/small molecule)
            f.write(f"show sticks, {structure_name} and organic\n")
            f.write(f"color gray60, {structure_name} and organic\n")

            # Note: Sticks for contributing atoms will be set after energy analysis

            # Add label
            f.write(f"label {structure_name} and name CA and resi 1, '{component}'\n")
            f.write(f"set label_size, 20\n")
            f.write(f"set label_color, black\n\n")

            # Get energy values for this component
            energy_values = protein_energy_differences[component]

            # Generate atom indices (1-based for PyMOL)
            atom_indices = list(range(1, n_atoms + 1))

            # Apply gradient energy-based coloring only to atoms within 10 Å of ligand
            f.write(f"# Gradient energy-based coloring for {component} component\n")
            f.write(f"# Set all atoms to gray first, then color significant energy atoms\n")

            # Create selection for atoms within 10 Å of ligand
            f.write(f"select near_ligand_{structure_name}, {structure_name} and (all within 10 of organic)\n")

            # Set all atoms within 10 Å to gray as base color (single command)
            f.write(f"color gray70, near_ligand_{structure_name}\n")

            # Calculate min/max for normalization within this component
            min_energy = float(np.min(energy_values))
            max_energy = float(np.max(energy_values))
            max_abs = max(abs(min_energy), abs(max_energy))

            # only color atoms with significant energy (>5% of max)
            energy_threshold = max_abs * 0.05

            # Only color atoms with significant energy values to optimize file size
            significant_atoms_colored = 0
            contributing_atoms = []  # Track atoms that contribute significantly to energy

            for i, (atom_idx, energy_val) in enumerate(zip(atom_indices, energy_values)):
                # Ensure energy_val is a float to avoid type errors
                try:
                    energy_val = float(energy_val)
                except (TypeError, ValueError):
                    logger.warning(f"Invalid energy value at atom {atom_idx}: {energy_val}, skipping")
                    continue

                if abs(energy_val) > energy_threshold:  # Only color significant energy atoms
                    contributing_atoms.append(atom_idx)

                    # Normalize energy value to [0, 1] based on magnitude
                    intensity = abs(energy_val) / max_abs

                    if energy_val > 0:
                        # Positive energy: gray → red gradient
                        r = 0.7 + 0.3 * intensity
                        g = 0.7 * (1 - intensity)
                        b = 0.7 * (1 - intensity)
                        color_name = f"red_{component}_{i}"
                    else:
                        # Negative energy: gray → blue gradient
                        r = 0.7 * (1 - intensity)
                        g = 0.7 * (1 - intensity)
                        b = 0.7 + 0.3 * intensity
                        color_name = f"blue_{component}_{i}"

                    # Define the gradient color and apply it
                    f.write(f"set_color {color_name}, [{r:.3f}, {g:.3f}, {b:.3f}]\n")
                    f.write(f"color {color_name}, {structure_name} and near_ligand_{structure_name} and id {atom_idx}\n")
                    significant_atoms_colored += 1

            # Show sticks for whole residues that contain contributing atoms
            if contributing_atoms:
                contributing_ids = "+".join(map(str, contributing_atoms))
                f.write(f"# Show sticks for whole residues containing contributing atoms\n")
                f.write(f"show sticks, {structure_name} and polymer and (byres (id {contributing_ids}))\n")

            f.write(f"# Colored {significant_atoms_colored} atoms with significant energy (>{energy_threshold:.4f})\n")

            # Add interaction detection commands
            f.write(f"# Detect and visualize interactions between ligand and protein\n")

            # Create selections for different types of interactions
            f.write(f"# Hydrogen bonds (3.2 Å cutoff)\n")
            f.write(f"select hbonds_{structure_name}, {structure_name} and polymer and (all within 3.2 of (organic and (elem N+O+S)))\n")
            f.write(f"distance hbond_dist_{structure_name}, {structure_name} and organic and (elem N+O+S), {structure_name} and polymer and (elem N+O+S), 3.2\n")

            f.write(f"# Hydrophobic contacts (4.5 Å cutoff for carbon atoms)\n")
            f.write(f"select hydrophobic_{structure_name}, {structure_name} and polymer and (elem C and all within 4.5 of (organic and elem C))\n")

            f.write(f"# Salt bridges/ionic interactions (4.0 Å cutoff)\n")
            f.write(f"select ionic_{structure_name}, {structure_name} and polymer and ((resn ARG+LYS+HIS and name NH*+NZ+ND1+NE2) or (resn ASP+GLU and name OD*+OE*)) and (all within 4.0 of organic)\n")

            f.write(f"# Pi-pi and cation-pi interactions (aromatic rings, 5.0 Å cutoff)\n")
            f.write(f"select aromatic_{structure_name}, {structure_name} and polymer and ((resn PHE and name CG+CD*+CE*+CZ) or (resn TYR and name CG+CD*+CE*+CZ+OH) or (resn TRP and name CG+CD*+NE1+CE*+CZ*+CH2) or (resn HIS and name CG+ND1+CD2+CE1+NE2)) and (all within 5.0 of organic)\n")

            # Style the interaction selections - show whole residues as sticks
            f.write(f"# Style interaction selections - show whole residues as sticks\n")
            f.write(f"show sticks, {structure_name} and polymer and (byres hbonds_{structure_name})\n")
            f.write(f"show sticks, {structure_name} and polymer and (byres hydrophobic_{structure_name})\n")
            f.write(f"show sticks, {structure_name} and polymer and (byres ionic_{structure_name})\n")
            f.write(f"show sticks, {structure_name} and polymer and (byres aromatic_{structure_name})\n")

            # Hide distance labels by default (can be shown manually)
            f.write(f"hide labels, hbond_dist_{structure_name}\n")
            f.write(f"color yellow, hbond_dist_{structure_name}\n")
            f.write(f"set dash_width, 2, hbond_dist_{structure_name}\n")

            f.write("\n\n")

        # Final setup
        f.write("# Final visualization setup\n")
        f.write("zoom all\n")
        f.write("set ray_opaque_background, off\n")
        f.write("set ray_trace_mode, 1\n")
        f.write("orient\n\n")

        # Create color scale legend info
        f.write("# Color scale information:\n")
        f.write("# Blue = Low energy values (normalized 0)\n")
        f.write("# Red = High energy values (normalized 1)\n")

        # Save session
        session_file = output_dir / f"{session_name}.pse"
        f.write(f"\n# Save session\n")
        f.write(f"save {session_file.name}\n")

    logger.info(f"PyMOL script generated. To visualize: pymol {script_path}")

    return str(script_path)