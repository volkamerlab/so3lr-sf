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

Structure Format Requirements:
- Preferred: PDB format for optimal 3D visualization and structural analysis
- PDB format provides proper residue identification, secondary structure, and atomic connectivity
- XYZ supported but may lack residues information and connectivity details

Energy Components Supported: MLFF, ZBL, Electrostatics, Dispersion, Total
Color Scheme: Gray (neutral) → Red (unfavorable) / Blue (favorable)
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Union

logger = logging.getLogger(__name__)


def add_interaction_detection(f, structure_name: str) -> None:
    """
    Add protein-ligand interaction detection commands to PyMOL script.

    This function generates PyMOL commands to detect and visualize various types of
    non-covalent interactions between a protein and ligand, including hydrogen bonds,
    hydrophobic contacts, ionic interactions, and aromatic interactions.

    Interaction Types Detected:

    1. Hydrogen Bonds (3.2 Å cutoff):
       - Detects potential H-bonds between polar atoms (N, O, S) in ligand and protein
       - Selection criteria: protein atoms within 3.2 Å of ligand N/O/S atoms
       - Visualization: yellow dashes showing distances between donor/acceptor pairs
       - Scientific basis: Typical H-bond lengths range from 2.5-3.2 Å

    2. Hydrophobic Contacts (4.5 Å cutoff):
       - Detects van der Waals interactions between carbon atoms
       - Selection criteria: protein carbon atoms within 4.5 Å of ligand carbon atoms
       - Visualization: protein residues shown as sticks to highlight hydrophobic patches
       - Scientific basis: van der Waals radii of C atoms (~1.7 Å) × 2 + interaction distance

    3. Salt Bridges/Ionic Interactions (4.0 Å cutoff):
       - Detects electrostatic interactions between charged groups
       - Protein cations: ARG (NH*, NZ), LYS (NZ), HIS (ND1, NE2)
       - Protein anions: ASP (OD*), GLU (OE*)
       - Selection criteria: charged protein atoms within 4.0 Å of organic ligand
       - Scientific basis: Ionic interactions effective up to ~4-5 Å in biological systems

    4. π-π and Cation-π Interactions (5.0 Å cutoff):
       - Detects aromatic ring interactions (π-π stacking, cation-π, π-cation)
       - Aromatic residues: PHE, TYR, TRP, HIS (aromatic ring atoms)
       - Selection criteria: aromatic protein atoms within 5.0 Å of organic ligand
       - Visualization: whole aromatic residues shown as sticks
       - Scientific basis: π-π interactions typically occur at 3.3-5.5 Å distances

    Visualization Features:
    - All interacting residues shown as sticks (whole residue, not just interacting atoms)
    - Hydrogen bond distances displayed as yellow dashes (hidden by default)
    - Dash width set to 2 for better visibility
    - Selections named with structure-specific prefixes to avoid conflicts

    Args:
        f: File handle for writing PyMOL commands
        structure_name: Name identifier for the protein structure in PyMOL

    Returns:
        None (writes commands directly to file)

    Example PyMOL selections created:
        - hbonds_{structure_name}: Protein atoms involved in H-bonds
        - hydrophobic_{structure_name}: Protein atoms in hydrophobic contacts
        - ionic_{structure_name}: Protein atoms in ionic interactions
        - aromatic_{structure_name}: Protein atoms in aromatic interactions
        - hbond_dist_{structure_name}: Distance measurements for H-bonds

    Note:
        This function assumes:
        - Protein is identified by 'polymer' selection
        - Ligand is identified by 'organic' selection
        - Structure follows standard PDB naming conventions
    """
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


def add_component_title(f, structure_name: str, component: str, center_of_mass: np.ndarray, x_offset: float) -> None:
    """
    Add a floating title label for an energy component above the structure.

    This function creates a pseudoatom positioned above the center of mass of a structure
    and adds a text label to identify the energy component being visualized. The title
    helps users distinguish between different energy components when multiple structures
    are displayed side by side.

    Title Positioning:
    - X coordinate: Center of mass X + horizontal offset (for side-by-side structures)
    - Y coordinate: Center of mass Y (no vertical offset)
    - Z coordinate: Center of mass Z + 50 Å (positioned above the structure)

    Styling Properties:
    - Label size: 25 (large enough to be clearly visible)
    - Label color: Black (high contrast against white background)
    - Pseudoatom: Small blue sphere (0.2 scale factor)
    - The pseudoatom serves as an anchor point for the floating label

    Args:
        f: File handle for writing PyMOL commands
        structure_name: Name identifier for the protein structure in PyMOL
        component: Name of the energy component (e.g., 'MLFF', 'Electrostatics', 'Total')
        center_of_mass: 3D coordinates [x, y, z] of the structure's center of mass
        x_offset: Horizontal translation offset for positioning multiple structures

    Returns:
        None (writes commands directly to file)

    Example PyMOL objects created:
        - title_{structure_name}: Pseudoatom positioned above structure center
        - Label text displaying the component name

    Note:
        The pseudoatom is styled as a small blue sphere to be minimally intrusive
        while providing a clear anchor point for the text label.
    """
    # Calculate title position above the center of this translated structure
    title_x = center_of_mass[0] + x_offset  # Center X + translation offset
    title_y = center_of_mass[1]             # Center Y (no Y translation)
    title_z = center_of_mass[2] + 50        # Center Z + height offset

    # Add energy component title using a pseudoatom positioned at structure center
    f.write(f"# Add floating title for {component} energy component at structure center\n")
    f.write(f"pseudoatom title_{structure_name}, pos=[{title_x:.3f}, {title_y:.3f}, {title_z:.3f}]\n")
    f.write(f"label title_{structure_name}, '{component}'\n")
    f.write(f"set label_size, 25, title_{structure_name}\n")
    f.write(f"set label_color, black, title_{structure_name}\n")
    f.write(f"show spheres, title_{structure_name}\n")
    f.write(f"set sphere_scale, 0.2, title_{structure_name}\n")
    f.write(f"color blue, title_{structure_name}\n")


def apply_energy_coloring(f, structure_name: str, component: str, atom_weights: Dict[str, float]) -> List[int]:
    """
    Apply energy-based gradient coloring to atoms and return contributing atom indices.

    This function implements a gray-to-color gradient coloring scheme based on energy values,
    where atoms with significant energy contributions are colored according to their magnitude
    and sign. Only atoms exceeding 5% of the maximum absolute energy are colored to focus
    on the most significant contributions.

    Coloring Algorithm:
    - Threshold: 5% of maximum absolute energy value
    - Positive energies: Gray → Red gradient (unfavorable interactions)
    - Negative energies: Gray → Blue gradient (favorable interactions)
    - Intensity scales linearly with energy magnitude
    - Base gray color: RGB(0.7, 0.7, 0.7)
    - Color transitions: Gray + 30% red/blue component based on intensity

    Color Calculation:
    - Positive: R = 0.7 + 0.3*intensity, G/B = 0.7*(1-intensity)
    - Negative: B = 0.7 + 0.3*intensity, R/G = 0.7*(1-intensity)
    - Intensity = |energy_value| / max_absolute_energy

    Args:
        f: File handle for writing PyMOL commands
        structure_name: Name identifier for the protein structure in PyMOL
        component: Energy component name (used for unique color naming)
        atom_weights: Dictionary mapping atom indices to energy values

    Returns:
        List[int]: Indices of atoms that received significant energy coloring
                  (atoms exceeding the 5% threshold)

    Example:
        >>> contributing_atoms = apply_energy_coloring(f, "protein_mlff", "MLFF",
        ...                                          {"1": -0.5, "2": 0.3, "3": -0.02})
        >>> print(contributing_atoms)  # [1, 2] (atom 3 below threshold)

    Note:
        - Invalid energy values are logged and skipped
        - Color names follow pattern: {red/blue}_{component}_{index}
        - Assumes atoms are within near_ligand_{structure_name} selection
    """
    # Filter out invalid energy values first
    valid_energy_values = []
    for idx, energy_val in atom_weights.items():
        try:
            valid_energy_values.append(float(energy_val))
        except (TypeError, ValueError):
            logger.warning(f"Invalid energy value encountered: {energy_val}, skipping atom {idx}")
            continue

    min_val, max_val = min(valid_energy_values), max(valid_energy_values)
    max_abs = max(abs(min_val), abs(max_val))

    # Only color atoms with significant energy (>5% of max)
    energy_threshold = max_abs * 0.05

    significant_atoms_colored = 0
    contributing_atoms = []  # Track atoms that contribute significantly to energy

    for i, (atom_idx, energy_val) in enumerate(atom_weights.items()):
        # Ensure energy_val is a float to avoid type errors
        try:
            energy_val = float(energy_val)
        except (TypeError, ValueError):
            logger.warning(f"Invalid energy value at atom {atom_idx}: {energy_val}, skipping")
            continue

        if abs(energy_val) > energy_threshold:  # Only color significant energy atoms
            contributing_atoms.append(int(atom_idx))

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

    f.write(f"# Colored {significant_atoms_colored} atoms with significant energy (>{energy_threshold:.4f})\n")

    return contributing_atoms


def create_pymol_session(complex_path: Union[str, Path],
                        atom_weights_mapped: Dict[str, List[float]],
                        output_path: Union[str, Path],
                        center_of_mass: np.ndarray) -> str:
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
        complex_path: Path to protein-ligand complex structure file
                     (PDB format preferred for optimal visualization; XYZ supported)
        protein_energy_differences: Dict with energy component names as keys
                                   and per-atom energy differences as values
                                   Components: MLFF, ZBL, Electrostatics, Dispersion, Total
        output_path: Path to save PyMOL session and script files
        session_name: Base name for output files (default: "protein_energy_visualization")

    Structure Format Notes:
        - PDB format recommended: Provides proper residue identification,
          secondary structure info, and atomic connectivity for optimal 3D visualization
        - XYZ format limitations: May lack residue information and connectivity details
          which can affect interaction detection and residue-based styling

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
        ...     'complex.pdb', energy_diffs, 'output_path', 'my_complex'
        ... )
        >>> print(f"Run: pymol {script_path}")
    """
    complex_path = Path(complex_path)

    # Validate input files
    if not complex_path.exists():
        raise FileNotFoundError(f"Protein file not found: {complex_path}")


    logger.info(f"Creating PyMOL visualization with "
                f"{len(atom_weights_mapped.keys())} energy components")

    # Calculate center of mass of the structure

    with open(output_path, 'w') as f:
        # Header
        f.write(f"# PyMOL Protein Energy Visualization\n")
        f.write(f"# Generated from: {complex_path}\n")
        f.write(f"# Components: {', '.join(atom_weights_mapped.keys())}\n\n")

        # Load and setup
        f.write("# Clear workspace\n")
        f.write("delete all\n")
        f.write("bg_color white\n\n")

        # Load protein structure for each component
        for i, (component, atom_weights) in enumerate(atom_weights_mapped.items()):
            structure_name = f"protein_{component.lower()}"
            f.write(f"# Load structure for {component} component\n")
            f.write(f"load {complex_path.name}, {structure_name}\n")

            # Set basic representation for protein-ligand complex
            f.write(f"# Hide everything first\n")
            f.write(f"hide everything, {structure_name}\n")

            # Show cartoon for protein (assuming protein is the first part of the complex)
            f.write(f"show cartoon, {structure_name} and polymer\n")
            f.write(f"color gray80, {structure_name} and polymer\n")

            # Show sticks for ligand (assuming ligand is organic/small molecule)
            f.write(f"show sticks, {structure_name} and organic\n")
            f.write(f"color gray60, {structure_name} and organic\n")

            # Set all atoms to gray then apply gradient energy-based coloring only to atoms within 10 Å of ligand
            f.write(f"# Gradient energy-based coloring for {component} component\n")
            f.write(f"# Set all atoms to gray first, then color significant energy atoms\n")

            # Create selection for atoms within 10 Å of ligand
            f.write(f"select near_ligand_{structure_name}, {structure_name} and (all within 10 of organic)\n")

            # Set all atoms within 10 Å to gray as base color (single command)
            f.write(f"color gray70, near_ligand_{structure_name}\n")

            # Apply energy-based coloring and get contributing atoms
            contributing_atoms = apply_energy_coloring(f, structure_name, component, atom_weights)

            # Show sticks for whole residues that contain contributing atoms
            if contributing_atoms:
                contributing_ids = "+".join(map(str, contributing_atoms))
                f.write(f"# Show sticks for whole residues containing contributing atoms\n")
                f.write(f"show sticks, {structure_name} and polymer and (byres (id {contributing_ids}))\n")

            # Delete the near_ligand selection to clean up
            f.write(f"delete near_ligand_{structure_name}\n")

            f.write("\n\n")

        # Add protein-ligand interaction using first component structure
        structure_name = f"protein_total"
        f.write("# Protein-ligand interaction detection (shared across all energy components)\n")
        add_interaction_detection(f, structure_name='protein_total')
        
        f.write("\n")

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
        session_file = output_path.with_suffix(".pse")
        f.write(f"\n# Save session\n")
        f.write(f"save {session_file.name}\n")

    logger.info(f"PyMOL script generated. To visualize: pymol {output_path}")

    return str(output_path)