"""
Explainability module for SO3LR-SF

This module provides explainability functionality for protein-ligand interactions
by calculating energy differences and generating molecular heatmaps.
"""

import io
import logging
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

from .molecule_loader import load_molecule_to_prolif
from .explain_utils import (compute_protein_ligand_interactions, get_atom_mappings, create_colorbar, similarity_map_gen, get_interaction_color,
                          add_interaction_summary, residue_weights_calculation)


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


def _augment_molecule_with_residues(
    rdkit_mol: Chem.Mol,
    residue_interactions: Dict[str, List[Dict[str, Any]]],
    residue_weights: Optional[Dict[str, Dict[str, float]]],
    component: str,
    base_weights: Sequence[float],
) -> Tuple[Chem.RWMol, List[float], List[int], Dict[int, Tuple[float, float, float]]]:
    """
    Attach residue pseudo-atoms and highlight bonds to the ligand copy.
    """
    lig_with_interactions = Chem.RWMol(rdkit_mol)
    highlight_bonds: List[int] = []
    highlight_bond_colors: Dict[int, Tuple[float, float, float]] = {}
    seen_bonds: set[int] = set()
    extended_weights = [float(weight) for weight in base_weights]
    original_atom_count = rdkit_mol.GetNumAtoms()

    for residue, interactions in residue_interactions.items():
        residue_atom = Chem.Atom(0)
        residue_atom.SetProp("atomLabel", residue)
        residue_idx = lig_with_interactions.AddAtom(residue_atom)

        component_energy = 0.0
        if residue_weights and residue in residue_weights:
            component_energy = float(residue_weights[residue].get(component, 0.0))
        extended_weights.append(component_energy)

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
    """
    rdDepictor.Compute2DCoords(molecule, bondLength=3.0, forceRDKit=True)
    drawer = rdMolDraw2D.MolDraw2DCairo(2400, 2400)
    draw_options = drawer.drawOptions()
    draw_options.circleAtoms = True
    draw_options.fillHighlights = True
    draw_options.continuousHighlight = False
    draw_options.highlightRadius = 0.5
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
        output_path = Path(output_path).with_stem(output_path.stem + "_ifp")
        fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
        logger.info(f"Protein interaction heatmap saved to: {output_path}")

    return fig


def generate_energy_heatmap(
    ligand_path: Union[str, Path],
    ligand_energy_differences: Dict[str, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    protein_energy_differences: Optional[Dict[str, np.ndarray]] = None,
    preloaded_protein_prolif: Optional[Tuple[plf.Molecule, Dict[str, List[int]]]] = None,
    total_interaction_energy: Optional[float] = None
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
    logger.info(f"Generating heatmap for ligand: {ligand_path}")
    # Read ligand molecule for visualization using universal function
    mol = load_molecule_to_prolif(ligand_path)

    # Use enhanced visualizer if protein data is provided (protein mode)
    if preloaded_protein_prolif is not None:
        atom_mappings = _fp_interaction_mapping(preloaded_protein_prolif, mol)
        residue_atom_mapping = preloaded_protein_prolif[1]  # Get residue mapping
        output_path = Path(output_path).with_stem(output_path.stem + "_ifp") if output_path else None
        # Use the protein interaction heatmap with proper bond coloring and interaction legends
        return generate_protein_interaction_heatmap(
            ligand_path=ligand_path,
            ligand_energy_differences=ligand_energy_differences,
            atom_mappings=atom_mappings,
            protein_energy_differences=protein_energy_differences,
            residue_atom_mapping=residue_atom_mapping,
            output_path=output_path,
            title=title,
            total_interaction_energy=total_interaction_energy
        )

    else:
        # Use basic ligand heatmap for non-protein mode
        return generate_ligand_heatmap(
            ligand_path=ligand_path,
            ligand_energy_differences=ligand_energy_differences,
            output_path=output_path,
            title=title,
            total_interaction_energy=total_interaction_energy
        )
