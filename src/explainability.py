"""
Explainability module for SO3LR-SF

This module provides explainability functionality for protein-ligand interactions
by calculating energy differences and generating molecular heatmaps.
"""

import io
import logging
from ase import Atoms
from ase.io import read
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple, List, Sequence
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import prolif as plf

logger = logging.getLogger(__name__)

from .utils import write_structure
from .molecule_loader import load_molecule_to_prolif, load_ase_structure
from .explain_utils import (compute_protein_ligand_interactions, get_atom_mappings, create_colorbar, similarity_map_gen, get_interaction_color,
                          add_interaction_summary, residue_weights_calculation, atom_weights_calculation)
from .visualizer import create_pymol_session


def _create_3d_energy_visualization(
    protein_atoms: Atoms,
    ligand_path: Union[str, Path],
    protein_energy_differences: Dict[str, np.ndarray],
    ligand_energy_differences: Dict[str, np.ndarray],
    output_path: Union[str, Path] = None
) -> Optional[str]:
    """
    Create 3D PyMOL visualization for protein-ligand complex with energy mapping.

    Args:
        protein_atoms: ASE Atoms object for protein structure
        ligand_path: Path to ligand structure file
        protein_energy_differences: Per-atom energy differences for protein
        ligand_energy_differences: Per-atom energy differences for ligand
        output_path: Optional output path for PyMOL script file

    Returns:
        Path to PyMOL script file if successful, None if failed
    """
    try:
        ligand_path = Path(ligand_path)

        ligand_name = ligand_path.stem
        logger.info("Creating 3D PyMOL visualization with mapped weights...")

        # Load ligand atoms for proper mapping
        ligand_atoms = load_ase_structure(ligand_path)[0]

        # Calculate properly mapped atom weights for 3D visualization
        atom_weights_mapped = atom_weights_calculation(
            protein_energy_differences=protein_energy_differences,
            ligand_energy_differences=ligand_energy_differences,
            protein_atoms=protein_atoms,
            ligand_atoms=ligand_atoms
        )

        logger.debug(f"Complex energy components: {list(atom_weights_mapped.keys())}")
        def concat_complex(protein: Atoms, ligand: Atoms, 
                        resname: str = 'LIG', chain: str = 'L') -> Atoms:
            """Concatenate protein and ligand, preserving PDB residue info."""
            n_lig = len(ligand)
            
            # Add matching arrays to ligand before concatenation
            if 'residuenames' in protein.arrays:
                ligand.set_array('residuenames', np.array([resname] * n_lig))
            
            if 'residuenumbers' in protein.arrays:
                max_resnum = protein.arrays['residuenumbers'].max()
                ligand.set_array('residuenumbers', np.array([max_resnum + 1] * n_lig))
            
            if 'chainids' in protein.arrays:
                ligand.set_array('chainids', np.array([chain] * n_lig))
            
            return protein + ligand
        # Create complex structure file in pdb format
        complex_path = output_path.parent / f"complex_{ligand_name}.pdb"
        # Load ligand and combine with protein
        ligand_atoms = load_ase_structure(ligand_path)[0]

        complex_atoms = concat_complex(protein_atoms, ligand_atoms)

        write_structure(complex_atoms, complex_path)
        if not complex_path.exists():
            raise FileNotFoundError(f"Failed to create complex file: {complex_path}")

        # Generate PyMOL visualization
        pml_file = create_pymol_session(
            complex_path=complex_path,
            atom_weights_mapped=atom_weights_mapped,
            output_path=output_path,
        )

        return pml_file

    except Exception as e:
        logger.error(f"Failed to create 3D visualization: {e}")
        return None


def _fp_interaction_mapping(preloaded_protein_prolif: plf.Molecule,
                           ligand_mol) -> List[Dict[str, Any]]:
    """
    Compute protein-ligand interaction fingerprint and extract atom mappings.

    Args:
        preloaded_protein_prolif: ProLIF protein molecule object
        ligand_mol: RDKit molecule object for ligand

    Returns:
        List of atom mapping dictionaries for protein-ligand interactions
    """
    fp = compute_protein_ligand_interactions(preloaded_protein_prolif[0], ligand_mol)
    atom_mappings = get_atom_mappings(fp)
    
    return atom_mappings


def _group_interactions_by_residue(
    atom_mappings: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Collect ligand interactions grouped by protein residue.
    """
    residue_interactions: Dict[str, List[Dict[str, Any]]] = {}
    for mapping in atom_mappings:
        residue = mapping.get("protein_residue", "Unknown")
        interaction = {
            "ligand_atoms": mapping.get("ligand_atoms", []),
            "interaction_type": mapping.get("interaction_type", "Unknown"),
        }
        residue_interactions.setdefault(residue, []).append(interaction)
    return residue_interactions


def _find_non_overlapping_position(
    target_x: float,
    target_y: float,
    existing_positions: List[Tuple[float, float]],
    conf,
    original_atom_count: int,
    min_distance: float = 2.5
) -> Tuple[float, float]:
    """
    Find a position near the target that doesn't overlap with existing positions or ligand atoms.
    Uses a spiral search pattern to find the best available position.
    """
    import math

    # Pre-compute ligand atom positions for faster access
    if not hasattr(_find_non_overlapping_position, '_ligand_cache'):
        _find_non_overlapping_position._ligand_cache = {}

    cache_key = id(conf)
    if cache_key not in _find_non_overlapping_position._ligand_cache:
        ligand_positions = []
        for atom_idx in range(original_atom_count):
            pos = conf.GetAtomPosition(atom_idx)
            ligand_positions.append((pos.x, pos.y))
        _find_non_overlapping_position._ligand_cache[cache_key] = ligand_positions

    ligand_positions = _find_non_overlapping_position._ligand_cache[cache_key]
    min_distance_sq = min_distance * min_distance  # Use squared distance to avoid sqrt

    def check_collision(x, y):
        # Check collision with existing pseudo atoms (squared distance)
        for ex_x, ex_y in existing_positions:
            dx, dy = x - ex_x, y - ex_y
            if dx*dx + dy*dy < min_distance_sq:
                return True

        # Check collision with cached ligand atoms (squared distance)
        for lig_x, lig_y in ligand_positions:
            dx, dy = x - lig_x, y - lig_y
            if dx*dx + dy*dy < min_distance_sq:
                return True

        return False

    # Try the target position first
    if not check_collision(target_x, target_y):
        return target_x, target_y

    max_radius = 6.0
    step_size = 0.3  # Larger step size
    angle_step = math.pi / 4 # 45 degrees - fewer angles

    radius = step_size
    while radius <= max_radius:
        angle = 0
        while angle < 2 * math.pi:
            test_x = target_x + radius * math.cos(angle)
            test_y = target_y + radius * math.sin(angle)

            if not check_collision(test_x, test_y):
                return test_x, test_y

            angle += angle_step
        radius += step_size

    # Final fallback: place at a guaranteed free position
    fallback_x = target_x + max_radius + 2.0
    fallback_y = target_y + max_radius + 2.0
    return fallback_x, fallback_y


def _augment_molecule_with_residues(
    rdkit_mol: Chem.Mol,
    residue_interactions: Dict[str, List[Dict[str, Any]]],
    residue_weights: Optional[Dict[str, Dict[str, float]]],
    component: str,
    base_weights: Sequence[float],
) -> Tuple[Chem.RWMol, List[float], List[int], Dict[int, Tuple[float, float, float]]]:
    """
    Attach residue pseudo-atoms and highlight bonds to the ligand copy.
    Preserves original ligand geometry by computing 2D coords first, then positioning pseudo atoms.
    """
    # First, compute 2D coordinates for the original ligand to fix its geometry
    rdDepictor.Compute2DCoords(rdkit_mol, bondLength=3.0, forceRDKit=True)

    lig_with_interactions = Chem.RWMol(rdkit_mol)
    highlight_bonds: List[int] = []
    highlight_bond_colors: Dict[int, Tuple[float, float, float]] = {}
    seen_bonds: set[int] = set()
    extended_weights = [float(weight) for weight in base_weights]
    original_atom_count = rdkit_mol.GetNumAtoms()

    # Get conformer to access and set coordinates
    conf = lig_with_interactions.GetConformer()

    # Track positions of placed pseudo atoms to avoid overlaps
    placed_pseudo_positions = []

    for residue, interactions in residue_interactions.items():
        residue_atom = Chem.Atom(0)
        residue_atom.SetProp("atomLabel", residue)
        residue_idx = lig_with_interactions.AddAtom(residue_atom)

        component_energy = 0.0
        if residue_weights and residue in residue_weights:
            component_energy = float(residue_weights[residue].get(component, 0.0))
        extended_weights.append(component_energy)

        # Calculate optimal position for pseudo atom based on connected ligand atoms
        connected_ligand_atoms = set()
        for interaction in interactions:
            ligand_atoms = interaction.get("ligand_atoms", [])
            for atom_idx in ligand_atoms:
                if atom_idx < original_atom_count:
                    connected_ligand_atoms.add(atom_idx)

        # Position pseudo atom near the center of connected ligand atoms
        if connected_ligand_atoms:
            # Calculate centroid of connected atoms
            x_sum = y_sum = 0.0
            for atom_idx in connected_ligand_atoms:
                pos = conf.GetAtomPosition(atom_idx)
                x_sum += pos.x
                y_sum += pos.y

            centroid_x = x_sum / len(connected_ligand_atoms)
            centroid_y = y_sum / len(connected_ligand_atoms)

            # Find a non-overlapping position using spiral placement
            pseudo_x, pseudo_y = _find_non_overlapping_position(
                centroid_x, centroid_y, placed_pseudo_positions, conf, original_atom_count
            )

            # Record this position to avoid future overlaps
            placed_pseudo_positions.append((pseudo_x, pseudo_y))

            # Set position for the pseudo atom
            from rdkit.Geometry import Point3D
            conf.SetAtomPosition(residue_idx, Point3D(pseudo_x, pseudo_y, 0.0))

        # Create bonds to interacting ligand atoms
        for interaction in interactions:
            ligand_atoms = interaction.get("ligand_atoms", [])
            if not ligand_atoms:
                continue

            bond_color = get_interaction_color(interaction.get("interaction_type"))
            for ligand_atom_idx in ligand_atoms:
                if ligand_atom_idx >= original_atom_count:
                    continue

                bond = lig_with_interactions.GetBondBetweenAtoms(residue_idx, ligand_atom_idx)
                if bond is None:
                    lig_with_interactions.AddBond(residue_idx, ligand_atom_idx, Chem.BondType.ZERO)
                    bond = lig_with_interactions.GetBondBetweenAtoms(residue_idx, ligand_atom_idx)
                if bond is None:
                    continue

                bond_idx = bond.GetIdx()
                if bond_idx in seen_bonds:
                    continue

                seen_bonds.add(bond_idx)
                highlight_bonds.append(bond_idx)
                highlight_bond_colors[bond_idx] = bond_color

    return lig_with_interactions, extended_weights, highlight_bonds, highlight_bond_colors


def _build_atom_colors(
    weights: Sequence[float],
    scale: float,
    colormap: Optional[Any] = None,
) -> Dict[int, Tuple[float, float, float]]:
    """
    Convert weights to color values for RDKit highlighting.
    """
    colormap = colormap or plt.get_cmap("bwr")
    safe_scale = max(scale, 1e-6)
    atom_colors: Dict[int, Tuple[float, float, float]] = {}

    for idx, weight in enumerate(weights):
        if np.isnan(weight):
            atom_colors[idx] = (0.8, 0.8, 0.8)
            logger.warning(f"Atom index {idx} has NaN weight, coloring as light gray.")
            continue

        normalized = max(-1.0, min(1.0, weight / safe_scale))
        color_value = (normalized + 1.0) / 2.0
        atom_colors[idx] = tuple(colormap(color_value)[:3])

    return atom_colors


def _draw_interaction_map(
    molecule: Chem.Mol,
    atom_colors: Dict[int, Tuple[float, float, float]],
    highlight_bonds: List[int],
    highlight_bond_colors: Dict[int, Tuple[float, float, float]],
) -> np.ndarray:
    """
    Render the augmented molecule with RDKit's 2D drawer.
    Assumes coordinates are already properly set and does not recompute them.
    """
    # Do NOT recompute coordinates - they are already set properly in _augment_molecule_with_residues
    drawer = rdMolDraw2D.MolDraw2DCairo(2400, 2400)
    draw_options = drawer.drawOptions()
    draw_options.circleAtoms = True
    draw_options.fillHighlights = True
    draw_options.continuousHighlight = False
    draw_options.highlightRadius = 0.8
    # Scale up line width and font sizes proportionally for higher resolution
    draw_options.bondLineWidth = 6
    draw_options.minFontSize = 28
    draw_options.maxFontSize = 38

    drawer.DrawMolecule(
        molecule,
        highlightAtoms=list(atom_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightBonds=highlight_bonds,
        highlightBondColors=highlight_bond_colors,
    )
    drawer.FinishDrawing()
    return np.asarray(Image.open(io.BytesIO(drawer.GetDrawingText())))

def generate_ligand_heatmap(
    ligand_path: Union[str, Path],
    ligand_energy_differences: Dict[str, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    total_interaction_energy: Optional[float] = None
) -> plt.Figure:
    """
    Generate basic ligand heatmap visualization for energy contributions.

    Args:
        ligand_path: Path to ligand structure file
        ligand_energy_differences: Per-atom energy differences for ligand atoms
        output_path: Optional path to save the figure
        title: Optional title for the figure

    Returns:
        plt.Figure: Generated matplotlib figure with ligand energy heatmaps
    """
    ligand_path = Path(ligand_path)

    # Read ligand molecule for visualization
    mol = load_molecule_to_prolif(ligand_path)

    # Create figure layout
    n_components = len(ligand_energy_differences)
    fig, axes = plt.subplots(1, n_components, figsize=(5*n_components, 5))

    if n_components == 1:
        axes = [axes]

    # Generate heatmap for each component
    for ax, (component, weights) in zip(axes, ligand_energy_differences.items()):
        global_max = float(np.max(np.abs(weights))) if len(weights) > 0 else 1.0

        if global_max == 0:
            global_max = 1e-6  # Avoid division by zero

        # Similarity map parameters
        kwargs = dict(
            sigma=0.5,
            gridResolution=0.02,
            contourLines=4,
            scale=global_max
        )
        rdDepictor.Compute2DCoords(mol)

        # Generate similarity map
        img = similarity_map_gen(mol, weights, cmap="bwr", width=700, height=700, **kwargs)

        # Plot image
        ax.imshow(img)
        ax.axis("off")

        # Add colorbar
        create_colorbar(fig, ax, global_max, component, weights)

    # Add main title with clarification about ligand-only contributions
    if title:
        # Calculate percentage of total interaction energy if provided
        if total_interaction_energy is not None and 'Total' in ligand_energy_differences:
            # Sum the ligand Total contributions
            ligand_total = float(np.sum(ligand_energy_differences['Total']))

            # Calculate percentage
            if abs(total_interaction_energy) > 1e-9:
                percentage = abs(ligand_total / total_interaction_energy) * 100
                clarified_title = f"{title}\n(Ligand Only: {percentage:.1f}% of Total Binding Energy)"
            else:
                clarified_title = f"{title}\n(Ligand Atom Energy Contributions Only)"
        else:
            clarified_title = f"{title}\n(Ligand Atom Energy Contributions Only)"

        fig.suptitle(clarified_title, fontsize=16, y=0.95)

    plt.tight_layout()

    # Save figure if requested
    if output_path:
        fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
        logger.info(f"Ligand heatmap saved to: {output_path}")

    return fig


def generate_protein_interaction_heatmap(
    ligand_path: Union[str, Path],
    ligand_energy_differences: Dict[str, np.ndarray],
    atom_mappings: List[Dict[str, Any]],
    protein_energy_differences: Optional[Dict[str, np.ndarray]] = None,
    residue_atom_mapping: Optional[Dict[str, List[int]]] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    total_interaction_energy: Optional[float] = None,
) -> plt.Figure:
    """
    Generate protein-ligand interaction heatmap visualization combining  interaction bonds and energy contributions for ligands with the interacting residues.

    Args:
        ligand_path: Path to ligand structure file
        ligand_energy_differences: Per-atom energy differences for ligand atoms
        atom_mappings: List of atom-level interaction mappings
        protein_energy_differences: Optional per-atom energy differences for protein atoms
        residue_atom_mapping: Optional mapping of residues to atom indices
        output_path: Optional path to save the figure
        title: Optional title for the figure
        total_interaction_energy: Optional total interaction energy for percentage calculations
        protein_atoms: Optional ASE Atoms object for protein structure. When provided, creates PyMOL session with 3D visualization of mapped weights

    Returns:
        plt.Figure: Generated matplotlib figure with energy heatmaps and interaction visualization
    """
    ligand_path = Path(ligand_path)

    mol = load_molecule_to_prolif(ligand_path)
    rdkit_mol = mol.mol if hasattr(mol, "mol") else mol

    residue_weights = None
    if protein_energy_differences is not None and residue_atom_mapping is not None:
        residue_weights = residue_weights_calculation(residue_atom_mapping, protein_energy_differences)
        
    residue_interactions = _group_interactions_by_residue(atom_mappings)
    rdDepictor.Compute2DCoords(rdkit_mol, bondLength=3.0)

    n_components = len(ligand_energy_differences)
    fig = plt.figure(figsize=(5 * n_components, 7))

    # Store extended weights for Total component for percentage calculation
    total_extended_weights = None
    ligand_energy_percent = None
    if total_interaction_energy is not None and 'Total' in ligand_energy_differences and abs(total_interaction_energy) > 1e-9:
        ligand_energy_percent = (sum(ligand_energy_differences['Total']) / total_interaction_energy) * 100
    for idx, (component, weights) in enumerate(ligand_energy_differences.items()):
        ax = fig.add_subplot(1, n_components, idx + 1)
        weights_array = np.asarray(weights, dtype=float)

        global_max = float(np.max(np.abs(weights_array))) if weights_array.size else 1.0
        if global_max == 0:
            global_max = 1e-6

        (
            lig_with_interactions,
            extended_weights,
            highlight_bonds,
            highlight_bond_colors,
        ) = _augment_molecule_with_residues(
            rdkit_mol,
            residue_interactions,
            residue_weights,
            component,
            weights_array,
        )

        # Store extended weights for Total component for percentage calculation
        if component == 'Total':
            total_extended_weights = extended_weights.copy()

        atom_colors = _build_atom_colors(extended_weights, global_max)
        ligand_heatmap_img = _draw_interaction_map(
            lig_with_interactions,
            atom_colors,
            highlight_bonds,
            highlight_bond_colors,
        )

        ax.imshow(ligand_heatmap_img)
        ax.axis("off")

        # Use extended_weights which includes both ligand and protein residue contributions
        create_colorbar(fig, ax, global_max, component, np.array(extended_weights))

    # Add main title with clarification about contributions included
    if title:
        # Calculate percentage of total interaction energy if provided
        if total_interaction_energy is not None and total_extended_weights is not None:
            # Sum the total extended weights (ligand + protein residues)
            displayed_total = float(np.sum(total_extended_weights))

            # Calculate percentage
            if abs(total_interaction_energy) > 1e-9:
                percentage = abs(displayed_total / total_interaction_energy) * 100
                if ligand_energy_percent is not None:
                    clarified_title = f"{title}\n(Total Binding Energy of the Ligand {ligand_energy_percent:.1f} + Interacting Residues: {percentage - ligand_energy_percent:.1f}%)"
                else:
                    clarified_title = f"{title}\n(Ligand + Interacting Residues: {percentage:.1f}% of Total Binding Energy)"
            else:
                clarified_title = f"{title}\n(Ligand + Interacting Protein Residue Contributions)"
        else:
            clarified_title = f"{title}\n(Ligand + Interacting Protein Residue Contributions)"

        fig.suptitle(clarified_title, fontsize=16, y=0.95)

    # Add interaction summary at bottom
    add_interaction_summary(fig, atom_mappings)

    plt.tight_layout()

    # Save figure if requested
    if output_path:
        fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
        logger.info(f"Protein interaction heatmap saved to: {output_path}")

    return fig


def generate_energy_heatmap(
    ligand_path: Union[str, Path],
    ligand_energy_differences: Dict[str, np.ndarray],
    output_paths: Tuple[Optional[Union[str, Path]], Optional[Union[str, Path]], Optional[Union[str, Path]]] = None,
    title: Optional[str] = None,
    protein_energy_differences: Optional[Dict[str, np.ndarray]] = None,
    preloaded_protein_prolif: Optional[Tuple[plf.Molecule, Dict[str, List[int]]]] = None,
    total_interaction_energy: Optional[float] = None,
    protein_atoms: Atoms = None,
) -> plt.Figure:
    """
    Generate heatmap visualization of ligand atom energy contributions.
    Enhanced to show protein-ligand interactions when protein data is provided.

    Args:
        ligand_path: Path to ligand structure file
        ligand_energy_differences: Per-atom energy differences for ligand atoms
        output_path: Optional path to save the figure
        title: Optional title for the figure
        protein_energy_differences: Optional per-atom energy differences for protein atoms
        preloaded_protein_prolif: Optional preloaded ProLIF protein data for enhanced visualization

    Returns:
        plt.Figure: Generated matplotlib figure with optional protein residue information
    """
    ligand_path = Path(ligand_path)
    logger.info(f"Generating heatmap for : {ligand_path}")
    # Read ligand molecule for visualization using universal function
    mol = load_molecule_to_prolif(ligand_path)
    # Use enhanced visualizer if protein data is provided (protein mode)
    # Create 3D PyMOL visualization if requested
    if protein_atoms is not None and protein_energy_differences is not None:
        _create_3d_energy_visualization(
            protein_atoms=protein_atoms,
            ligand_path=ligand_path,
            protein_energy_differences=protein_energy_differences,
            ligand_energy_differences=ligand_energy_differences,
            output_path=output_paths[-1]
        )

    # Generate ligand heatmap if requested or as fallback
    ligand_fig = None
    if output_paths[0] is not None:
        ligand_fig = generate_ligand_heatmap(
            ligand_path=ligand_path,
            ligand_energy_differences=ligand_energy_differences,
            output_path=output_paths[0],
            title=title,
            total_interaction_energy=total_interaction_energy
        )
        # Close immediately after saving to free memory
        plt.close(ligand_fig)

    # Generate protein interaction heatmap if requested and data available
    if output_paths[1] is not None and preloaded_protein_prolif is not None:
        atom_mappings = _fp_interaction_mapping(preloaded_protein_prolif, mol)
        residue_atom_mapping = preloaded_protein_prolif[1]
        protein_fig = generate_protein_interaction_heatmap(
            ligand_path=ligand_path,
            ligand_energy_differences=ligand_energy_differences,
            atom_mappings=atom_mappings,
            protein_energy_differences=protein_energy_differences,
            residue_atom_mapping=residue_atom_mapping,
            output_path=output_paths[1],
            title=title,
            total_interaction_energy=total_interaction_energy,
        )
        # Return the protein figure if it's the last/only one generated
        return protein_fig

    # If no output paths provided, generate and return basic ligand heatmap for testing
    if output_paths[0] is None and output_paths[1] is None:
        return generate_ligand_heatmap(
            ligand_path=ligand_path,
            ligand_energy_differences=ligand_energy_differences,
            output_path=None,
            title=title,
            total_interaction_energy=total_interaction_energy
        )

    # Return None since all figures are saved to files
    return None
