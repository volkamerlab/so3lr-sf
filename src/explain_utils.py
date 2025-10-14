"""
Visual utilities for SO3LR-SF explainability and heatmap generation.

This module contains shared visualization functions used across multiple
explainability modules for consistent styling and functionality.
"""

import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import prolif as plf
from typing import Dict, Any, Optional, Tuple, List
from matplotlib.colors import Normalize
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D


########################## GENERAL VISUALIZATION UTILITIES #########################
def create_colorbar(fig, ax, global_max: float, component: str, weights: np.ndarray) -> None:
    """
    Create and format colorbar for heatmap visualization.

    Args:
        fig: Matplotlib figure
        ax: Matplotlib axes
        global_max: Maximum absolute value for color scale
        component: Component name for label
        weights: Weight values for total calculation
    """
    sm = plt.cm.ScalarMappable(
        cmap="bwr",
        norm=Normalize(vmin=-global_max, vmax=global_max)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.08)

    # Format colorbar label
    component_total = np.sum(weights)
    if component == 'Total':
        cbar.set_label(f"{component} Energy = {component_total:.4f} eV",
                      fontsize=10, fontweight='bold')
    else:
        cbar.set_label(f"{component} total = {component_total:.4f} eV",
                      fontsize=10, fontweight='bold')


def similarity_map_gen(mol, weights, cmap="bwr", width=600, height=600, **kwargs):
    """
    Generate a similarity map image as a numpy array for ligand explainability.

    Args:
        mol: RDKit molecule object
        weights: List of weights for each atom
        cmap: Colormap name
        width: Image width
        height: Image height
        **kwargs: Additional similarity map parameters

    Returns:
        np.ndarray: Generated similarity map image
    """
    d2d = Draw.MolDraw2DCairo(width, height)

    if len(weights) != mol.GetNumAtoms():
        raise ValueError(
            f"Weights length {len(weights)} does not match number of atoms "
            f"in molecule {mol.GetNumAtoms()}"
        )

    _ = SimilarityMaps.GetSimilarityMapFromWeights(
        mol,
        list(map(float, weights)),
        colorMap=plt.get_cmap(cmap),
        draw2d=d2d,
        **kwargs
    )

    d2d.FinishDrawing()
    img_pil = Image.open(io.BytesIO(d2d.GetDrawingText()))
    return np.asarray(img_pil)

######################### FUNCTIONS FOR PROTEIN-LIGAND EXPLAINABILITY #########################

def similarity_map_with_colored_bonds(
    mol, weights, highlight_bonds, highlight_bond_colors,
    cmap="bwr", width=600, height=600, **kwargs
):
    """
    Generate a similarity map with custom bond coloring using rdMolDraw2D for protein-ligand interactions.

    Args:
        mol: RDKit molecule object
        weights: List of weights for each atom
        highlight_bonds: List of bond indices to highlight
        highlight_bond_colors: Dict mapping bond indices to RGB colors
        cmap: Colormap name
        width: Image width
        height: Image height
        **kwargs: Additional similarity map parameters

    Returns:
        np.ndarray: Generated similarity map image with colored bonds
    """
    if len(weights) != mol.GetNumAtoms():
        raise ValueError(
            f"Weights length {len(weights)} does not match number of atoms "
            f"in molecule {mol.GetNumAtoms()}"
        )

    # Create drawer with proper settings for normal atom sizes
    d2d = rdMolDraw2D.MolDraw2DCairo(width, height)
    opts = d2d.drawOptions()

    # Configure drawing options for proper atom sizing
    opts.circleAtoms = True
    opts.fillHighlights = True
    opts.continuousHighlight = False
    opts.highlightRadius = 0.5  # Larger radius for normal atom size
    opts.bondLineWidth = 3
    opts.minFontSize = 12
    opts.maxFontSize = 18

    # Create atom colors based on weights for heatmap effect
    atom_colors = {}
    colormap = plt.get_cmap(cmap)
    scale = kwargs.get('scale', 1.0)

    for i, weight in enumerate(weights):
        if np.isnan(weight):
            continue  # Skip nan weights
        normalized_weight = max(-1.0, min(1.0, weight / scale))  # Normalize to [-1, 1]
        color_val = (normalized_weight + 1.0) / 2.0  # Convert to [0, 1]
        rgb = colormap(color_val)[:3]  # Get RGB, ignore alpha
        atom_colors[i] = rgb

    # Draw molecule with both atom colors (for heatmap) and bond colors (for interactions)
    d2d.DrawMolecule(
        mol,
        highlightAtoms=list(atom_colors.keys()),
        highlightBonds=highlight_bonds,
        highlightBondColors=highlight_bond_colors,
        highlightAtomColors=atom_colors,
    )

    d2d.FinishDrawing()
    img_bytes = d2d.GetDrawingText()
    img_pil = Image.open(io.BytesIO(img_bytes))
    return np.asarray(img_pil)


def categorize_interaction(interaction_type: str) -> str:
    """
    Categorize an interaction type into broader categories.

    Args:
        interaction_type: The specific interaction type from ProLIF

    Returns:
        str: Category name matching the color scheme
    """
    interaction_lower = interaction_type.lower()

    if interaction_lower in ['hbdonor']:
        return 'HBDonor'
    elif interaction_lower in ['hbacceptor']:
        return 'HBAcceptor'
    elif interaction_lower in ['hbond', 'hydrophilic', 'h-bond']:
        return 'H-bond'
    elif interaction_lower in ['xbdonor', 'xbacceptor', 'halogen']:
        return 'Halogen bond'
    elif interaction_lower in ['hydrophobic']:
        return 'Hydrophobic'
    elif interaction_lower in ['vdwcontact', 'vdw', 'vanderwaals']:
        return 'VdW'
    elif interaction_lower in ['pistacking', 'pication', 'cationpi', 'edgetoface', 'facetoface', 'basepistacking', 'pi-stacking', 'pi']:
        return 'π-interaction'
    elif interaction_lower in ['anionic', 'cationic', 'saltbridge', 'ionic', 'salt-bridge']:
        return 'Ionic'
    elif interaction_lower in ['metaldonor', 'metalacceptor', 'metal']:
        return 'Metal'
    elif interaction_lower in ['distance', 'singleangle', 'doubleangle', 'angle']:
        return 'Distance/Angle'
    else:
        return 'Other'


def get_interaction_color(interaction_type: str) -> Tuple[float, float, float]:
    """
    Get color for single interaction type.

    Color scheme:
    - Blue: H-bonds (HBDonor, HBAcceptor)
    - Dark Blue: Halogen bonds (XBDonor, XBAcceptor)
    - Green: Hydrophobic and VdW contacts
    - Yellow: Pi interactions (PiStacking, PiCation, CationPi, EdgeToFace, FaceToFace, BasePiStacking)
    - Cyan: Ionic (Anionic, Cationic)
    - Purple: Metal interactions (MetalDonor, MetalAcceptor)
    - Gray: Distance/Angle measurements

    Args:
        interaction_type: Single interaction type

    Returns:
        tuple: RGB color as (R, G, B) with values 0-1
    """
    # Normalize interaction type to lowercase for comparison
    interaction_lower = interaction_type.lower()

    # Priority order: HBDonor > HBAcceptor > H-bonds > Halogen > Hydrophobic > VdW > Pi > Ionic > Metal > Distance/Angle
    if 'hbdonor' in interaction_lower:
        return (0.4, 0.6, 1.0)  # Light Blue
    elif 'hbacceptor' in interaction_lower:
        return (0.2, 0.4, 0.9)  # Medium Blue
    elif any(x in interaction_lower for x in ['hbond', 'hydrophilic', 'h-bond']):
        return (0.1, 0.3, 0.7)  # Dark Blue
    elif any(x in interaction_lower for x in ['xbdonor', 'xbacceptor', 'halogen']):
        return (0.1, 0.2, 0.5)  # Very Dark Blue
    elif 'hydrophobic' in interaction_lower:
        return (0.1, 0.6, 0.1)  # Dark Green
    elif any(x in interaction_lower for x in ['vdwcontact', 'vdw', 'vanderwaals']):
        return (0.5, 0.9, 0.5)  # Light Green
    elif any(x in interaction_lower for x in ['pistacking', 'pication', 'cationpi', 'edgetoface', 'facetoface', 'basepistacking', 'pi-stacking', 'pi']):
        return (0.9, 0.9, 0.2)  # Yellow
    elif any(x in interaction_lower for x in ['anionic', 'cationic', 'saltbridge', 'ionic', 'salt-bridge']):
        return (0.3, 0.8, 0.8)  # Cyan
    elif any(x in interaction_lower for x in ['metaldonor', 'metalacceptor', 'metal']):
        return (0.8, 0.2, 0.8)  # Purple/Magenta
    elif any(x in interaction_lower for x in ['distance', 'singleangle', 'doubleangle', 'angle']):
        return (0.6, 0.6, 0.6)  # Gray
    else:
        return (0.5, 0.5, 0.5)  # Default gray


def add_interaction_summary(fig: plt.Figure, atom_mappings: List[Dict[str, Any]]) -> None:
    """Add interaction summary with colored markers, displayed horizontally."""
    # Group interactions by category
    category_counts = {}
    for mapping in atom_mappings:
        category = categorize_interaction(mapping['interaction_type'])
        category_counts[category] = category_counts.get(category, 0) + 1

    # Color mapping for categories (matching the legend)
    color_map = {
        'HBDonor': (0.4, 0.6, 1.0),
        'HBAcceptor': (0.2, 0.4, 0.9),
        'H-bond': (0.1, 0.3, 0.7),
        'Halogen bond': (0.1, 0.2, 0.5),
        'Hydrophobic': (0.1, 0.6, 0.1),
        'VdW': (0.5, 0.9, 0.5),
        'π-interaction': (0.9, 0.9, 0.2),
        'Ionic': (0.3, 0.8, 0.8),
        'Metal': (0.8, 0.2, 0.8),
        'Distance/Angle': (0.6, 0.6, 0.6),
        'Other': (0.5, 0.5, 0.5)
    }

    # Calculate total width to center the entire summary
    total_categories = len(category_counts)
    if total_categories == 0:
        return

    # Build all text parts and calculate total width more accurately
    text_parts = []
    total_width = 0

    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        text = f'{category} ({count})'
        text_parts.append((category, count, text))
        # More accurate width calculation: marker + space + text + spacing
        total_width += 0.03 + len(text) * 0.009 + 0.04  # marker + text + spacing

    # Start position to center the entire content beneath the title
    x_start = 0.5 - (total_width / 2)
    y_pos = 0.85  # More centered between title and heatmaps
    x_offset = 0.0

    for category, count, text in text_parts:
        color = color_map.get(category, (0.5, 0.5, 0.5))

        # Add colored circle marker
        fig.text(x_start + x_offset, y_pos, '●', transform=fig.transFigure,
                fontsize=16, color=color, verticalalignment='center')

        # Add category text
        fig.text(x_start + x_offset + 0.02, y_pos, text,
                transform=fig.transFigure, fontsize=12, verticalalignment='center')

        # Move to next position
        text_width = 0.02 + len(text) * 0.007 + 0.03
        x_offset += text_width


############################# UTILITIES FOR PROTEIN-LIGAND EXPLAINABILITY #########################

def compute_protein_ligand_interactions(
    protein_mol: plf.Molecule,
    ligand_mol: plf.Molecule,
) -> plf.Fingerprint:
    """
    Compute protein-ligand interactions using ProLIF fingerprinting.

    Args:
        protein_mol: ProLIF protein molecule
        ligand_mol: ProLIF ligand molecule
        interactions: List of interaction types to compute. If None, uses default set.

    Returns:
        plf.Fingerprint: Fingerprint object with atom-level interaction data
                         Use .to_dataframe() for summary or .ifp for atom mappings
    """

    # Create fingerprint generator
    fp = plf.Fingerprint()

    # Compute interactions
    fp.run_from_iterable([ligand_mol], protein_mol)

    return fp  # Return fingerprint object with full atom mapping data


def get_atom_mappings(fp: plf.Fingerprint) -> List[Dict[str, Any]]:
    """
    Extract atom-level mappings from ProLIF fingerprint.

    Args:
        fp: ProLIF Fingerprint object after running

    Returns:
        list: List of interaction details with atom indices
    """
    mappings = []

    for frame_idx, frame_data in fp.ifp.items():
        for (lig_res, prot_res), interaction_dict in frame_data.items():
            for interaction_type, metadata_tuple in interaction_dict.items():
                if metadata_tuple:
                    # metadata_tuple is a tuple containing dict(s)
                    metadata = metadata_tuple[0] if metadata_tuple else {}

                    mapping = {
                        'frame': frame_idx,
                        'ligand_residue': str(lig_res),
                        'protein_residue': str(prot_res),
                        'interaction_type': interaction_type,
                    }

                    # Extract indices
                    if 'indices' in metadata:
                        indices = metadata['indices']
                        mapping['ligand_atoms'] = list(indices.get('ligand', ()))
                        mapping['protein_atoms'] = list(indices.get('protein', ()))

                    # Extract additional info with rounding
                    if 'distance' in metadata:
                        mapping['distance'] = round(metadata['distance'], 3)
                    if 'DHA_angle' in metadata:
                        mapping['DHA_angle'] = round(metadata['DHA_angle'], 3)

                    mappings.append(mapping)

    return mappings

def residue_weights_calculation(
    residue_atom_indices: Dict[str, List[int]],
    protein_energy_differences: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate residue-level energy contributions from atom-level mappings.

    Args:
        residue_atom_indices: Dictionary mapping residue names to atom indices
        protein_energy_differences: Per-atom energy differences for protein atoms

    Returns:
        dict: Residue-level energy contributions by component
    """
    residue_weights = {}

    for residue, atom_indices in residue_atom_indices.items():
        if residue not in residue_weights:
            residue_weights[residue] = {comp: 0.0 for comp in protein_energy_differences.keys()}

        for comp, values in protein_energy_differences.items():
            # Calculate sum energy for this residue's atoms (not mean)
            residue_energies = [values[idx] for idx in atom_indices if idx < len(values)]
            if residue_energies:
                residue_weights[residue][comp] = float(np.sum(residue_energies))
            else:
                residue_weights[residue][comp] = 0.0

    # Calculate total contribution per residue
    for residue, comp_dict in residue_weights.items():
        comp_dict['Total'] = sum(comp_dict.values())

    return residue_weights

def compute_energy_differences(
    protein_components: Dict[str, np.ndarray],
    ligand_components: Dict[str, np.ndarray],
    complex_components: Dict[str, np.ndarray],
    n_protein_atoms: int,
    n_ligand_atoms: int,
    protein_mode: bool = False
) -> Tuple[Dict[str, List[float]], Optional[Dict[str, List[float]]]]:
    """
    Compute per-atom energy differences for ligand and optionally protein atoms.

    Calculates interaction energy differences by comparing atoms in the complex
    vs. their isolated states: complex_part - isolated for each energy component.
    Maps internal component names to user-friendly names (e.g., 'mlff_atomic_energy' -> 'MLFF').

    Args:
        protein_components: Per-atom energy components for isolated protein
        ligand_components: Per-atom energy components for isolated ligand
        complex_components: Per-atom energy components for protein-ligand complex
        n_protein_atoms: Number of protein atoms
        n_ligand_atoms: Number of ligand atoms
        protein_mode: If True, also compute protein energy differences

    Returns:
        Tuple of (ligand_differences, protein_differences):
        - ligand_differences: Dict mapping component names to per-atom energy differences for ligand
        - protein_differences: Dict mapping component names to per-atom energy differences for protein
          (None if protein_mode=False)
    """
    # Component mapping to standard names
    component_mapping = {
        'mlff_atomic_energy': 'MLFF',
        'zbl_repulsion': 'ZBL',
        'electrostatic_energy': 'Electrostatics',
        'dispersion_energy': 'Dispersion'
    }
    protein_differences = {}
    ligand_differences = {}

    for original_comp, mapped_comp in component_mapping.items():
        if (original_comp in protein_components and
            original_comp in ligand_components and
            original_comp in complex_components):

            # Get ligand part from complex (atoms after protein atoms)
            complex_ligand_part = complex_components[original_comp][n_protein_atoms:n_protein_atoms + n_ligand_atoms]
            ligand_alone = ligand_components[original_comp]

            # Calculate difference: complex_ligand - ligand_alone
            ligand_diff = complex_ligand_part - ligand_alone
            ligand_differences[mapped_comp] = ligand_diff
            if protein_mode:
                # calculate protein energy difference
                protein_part = complex_components[original_comp][:n_protein_atoms]
                protein_alone = protein_components[original_comp]
                protein_diff = protein_part - protein_alone
                protein_differences[mapped_comp] = protein_diff

    # Calculate total as sum of all components
    if ligand_differences:
        ligand_differences['Total'] = sum(ligand_differences.values())
        ligand_dict = {comp: values.tolist() for comp, values in ligand_differences.items()}

        if protein_differences and protein_mode:
            protein_differences['Total'] = sum(protein_differences.values())
            protein_dict = {comp: values.tolist() for comp, values in protein_differences.items()}
            return ligand_dict, protein_dict
        else:
            return ligand_dict, None

    return {}, None