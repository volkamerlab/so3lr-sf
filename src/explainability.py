"""
Explainability module for SO3LR-SF

This module provides explainability functionality for protein-ligand interactions
by calculating energy differences and generating molecular heatmaps.
"""

import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple, List
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import prolif as plf

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

def generate_ligand_heatmap(
    ligand_path: Union[str, Path],
    ligand_energy_differences: Dict[str, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None
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
    print(f"Generating ligand heatmap for: {ligand_path}")

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

    # Add main title
    if title:
        fig.suptitle(title, fontsize=16, y=0.95)

    plt.tight_layout()

    # Save figure if requested
    if output_path:
        fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"Ligand heatmap saved to: {output_path}")

    return fig


def generate_protein_interaction_heatmap(
    ligand_path: Union[str, Path],
    ligand_energy_differences: Dict[str, np.ndarray],
    atom_mappings: List[Dict[str, Any]],
    protein_energy_differences: Optional[Dict[str, np.ndarray]] = None,
    residue_atom_mapping: Optional[Dict[str, List[int]]] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
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
    print(f"Generating protein interaction heatmap for ligand: {ligand_path}")

    # Load ligand molecule
    mol = load_molecule_to_prolif(ligand_path)
    rdkit_mol = mol.mol if hasattr(mol, 'mol') else mol

    # Calculate residue weights if protein energy differences are available
    residue_weights = None
    if protein_energy_differences is not None and residue_atom_mapping is not None:
        residue_weights = residue_weights_calculation(
            residue_atom_mapping, protein_energy_differences
        )

    # Create figure for integrated ligand-residue visualization
    n_components = len(ligand_energy_differences)
    fig = plt.figure(figsize=(5*n_components, 7))

    # Generate ligand heatmaps with integrated residue interactions
    for i, (component, weights) in enumerate(ligand_energy_differences.items()):
        ax = fig.add_subplot(1, n_components, i + 1)

        global_max = float(np.max(np.abs(weights))) if len(weights) > 0 else 1.0
        if global_max == 0:
            global_max = 1e-6

        print(f"{component}: max absolute contribution = {global_max:.6f} eV")

        # Create molecule with residue pseudo-atoms for this component
        lig_with_interactions = Chem.RWMol(rdkit_mol)

        # Build visualization elements
        highlight_bonds = []
        highlight_bond_colors = {}
        seen_bonds = set()
        extended_weights = list(weights)  # Start with original ligand weights

        # Process each interaction mapping to add pseudo-atoms and bonds
        for mapping in atom_mappings:
            residue = mapping.get('protein_residue', 'Unknown')
            interaction_type = mapping.get('interaction_type', 'Unknown')
            ligand_atoms = mapping.get('ligand_atoms', [])

            # Create residue pseudo-atom with proper label
            res_atom = Chem.Atom(0)  # Dummy atom
            res_atom.SetProp('atomLabel', residue)
            res_idx = lig_with_interactions.AddAtom(res_atom)

            # Add weight for this residue pseudo-atom to extended weights
            if residue_weights and residue in residue_weights:
                # Use the component-specific energy for this residue as its weight
                component_energy = residue_weights[residue].get(component, 0.0)
                extended_weights.append(component_energy)
            else:
                # Default weight of 0 for residues without energy data
                extended_weights.append(0.0)

            # Get bond color based on interaction type
            bond_color = get_interaction_color(interaction_type)

            # Add colored bonds from residue pseudo-atom to ligand atoms
            for ligand_atom_idx in ligand_atoms:
                if ligand_atom_idx < rdkit_mol.GetNumAtoms():  # Use original molecule atom count
                    lig_with_interactions.AddBond(res_idx, ligand_atom_idx, Chem.BondType.ZERO)
                    bond = lig_with_interactions.GetBondBetweenAtoms(res_idx, ligand_atom_idx)

                    if bond is not None:
                        bond_idx = bond.GetIdx()
                        if bond_idx not in seen_bonds:
                            seen_bonds.add(bond_idx)
                            highlight_bonds.append(bond_idx)
                            highlight_bond_colors[bond_idx] = bond_color

        # Generate 2D coordinates for clean 2D visualization
        rdDepictor.Compute2DCoords(lig_with_interactions, bondLength=3.0)

        # Create unified heatmap with both ligand atoms and residue pseudo-atoms
        similarity_kwargs = dict(
            sigma=0.5,
            gridResolution=0.02,
            contourLines=4,
            scale=global_max
        )

        # Use rdMolDraw2D directly with higher resolution
        d2d = rdMolDraw2D.MolDraw2DCairo(1200, 1200)  # Higher resolution
        opts = d2d.drawOptions()

        # Configure drawing options back to normal
        opts.circleAtoms = True
        opts.fillHighlights = True
        opts.continuousHighlight = False
        opts.highlightRadius = 0.5       # Normal highlight radius
        opts.bondLineWidth = 3
        opts.minFontSize = 12
        opts.maxFontSize = 18

        # Create atom colors based on weights for heatmap effect
        atom_colors = {}
        colormap = plt.get_cmap("bwr")
        scale = similarity_kwargs.get('scale', 1.0)

        for i, weight in enumerate(extended_weights):
            if np.isnan(weight):
                # Use neutral color for NaN weights
                atom_colors[i] = (0.8, 0.8, 0.8)  # Light gray
                print(f"Warning: Atom index {i} has NaN weight, coloring as light gray.")
                continue

            # Use same scale for all atoms (ligand and pseudo-atoms)
            normalized_weight = max(-1.0, min(1.0, weight / scale))
            color_val = (normalized_weight + 1.0) / 2.0
            rgb = colormap(color_val)[:3]

            # Debug pseudo-atoms (indices beyond original ligand)
            if i >= rdkit_mol.GetNumAtoms():
                print(f"Pseudo-atom {i}: weight={weight:.6f}, normalized={normalized_weight:.6f}, color_val={color_val:.3f}, rgb={rgb}")

            atom_colors[i] = rgb

        # Draw molecule with both atom colors (heatmap) and bond colors (interactions)
        d2d.DrawMolecule(
            lig_with_interactions,
            highlightAtoms=list(atom_colors.keys()),
            highlightBonds=highlight_bonds,
            highlightAtomColors=atom_colors,
            highlightBondColors=highlight_bond_colors
        )

        d2d.FinishDrawing()
        ligand_heatmap_img = np.asarray(Image.open(io.BytesIO(d2d.GetDrawingText())))

        # Plot the unified visualization
        ax.imshow(ligand_heatmap_img)
        ax.axis("off")

        # Add colorbar using shared helper
        create_colorbar(fig, ax, global_max, component, weights)

    # Add main title
    if title:
        fig.suptitle(title, fontsize=16, y=0.95)

    # Add interaction summary at bottom
    add_interaction_summary(fig, atom_mappings)

    plt.tight_layout()

    # Save figure if requested
    if output_path:
        fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"Protein interaction heatmap saved to: {output_path}")

    return fig


def generate_energy_heatmap(
    ligand_path: Union[str, Path],
    ligand_energy_differences: Dict[str, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    protein_energy_differences: Optional[Dict[str, np.ndarray]] = None,
    preloaded_protein_prolif: Optional[Tuple[plf.Molecule, Dict[str, List[int]]]] = None
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
    print(f"Generating heatmap for ligand: {ligand_path}")
    # Read ligand molecule for visualization using universal function
    mol = load_molecule_to_prolif(ligand_path)

    # Use enhanced visualizer if protein data is provided (protein mode)
    if preloaded_protein_prolif is not None:
        atom_mappings = _fp_interaction_mapping(preloaded_protein_prolif, mol)
        residue_atom_mapping = preloaded_protein_prolif[1]  # Get residue mapping

        # Use the protein interaction heatmap with proper bond coloring and interaction legends
        return generate_protein_interaction_heatmap(
            ligand_path=ligand_path,
            ligand_energy_differences=ligand_energy_differences,
            atom_mappings=atom_mappings,
            protein_energy_differences=protein_energy_differences,
            residue_atom_mapping=residue_atom_mapping,
            output_path=output_path,
            title=title
        )

    else:
        # Use basic ligand heatmap for non-protein mode
        return generate_ligand_heatmap(
            ligand_path=ligand_path,
            ligand_energy_differences=ligand_energy_differences,
            output_path=output_path,
            title=title
        )