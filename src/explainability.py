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
from typing import Dict, Any, Union, Optional, Tuple
from matplotlib.colors import Normalize
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps


from .molecule_loader import load_molecule_to_prolif


def _similarity_map_gen(mol, weights, cmap="bwr", width=600, height=600, **kwargs):
    """
    Generate a similarity map image as a numpy array.

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


def generate_interaction_heatmap(
    ligand_path: Union[str, Path],
    ligand_energy_differences: Dict[str, np.ndarray],
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """
    Generate heatmap visualization of ligand atom energy contributions.

    Args:
        ligand_path: Path to ligand structure file
        ligand_energy_differences: Per-atom energy differences for ligand atoms
        output_path: Optional path to save the figure
        title: Optional title for the figure

    Returns:
        plt.Figure: Generated matplotlib figure
    """
    ligand_path = Path(ligand_path)
    print(f"Generating heatmap for ligand: {ligand_path}")
    # Read ligand molecule for visualization using universal function
    mol = load_molecule_to_prolif(ligand_path)

    if mol is None:
        raise ValueError(f"Could not read molecule from {ligand_path}")

    # Setup subplot grid
    n_components = len(ligand_energy_differences)
    fig, axes = plt.subplots(1, n_components, figsize=(5*n_components, 5))

    if n_components == 1:
        axes = [axes]

    # Generate heatmap for each component
    for ax, (component, weights) in zip(axes, ligand_energy_differences.items()):
        global_max = float(np.max(np.abs(weights))) if len(weights) > 0 else 1.0

        if global_max == 0:
            global_max = 1e-6  # Avoid division by zero

        print(f"{component}: max absolute contribution = {global_max:.6f} eV")

        # Similarity map parameters
        kwargs = dict(
            sigma=0.5,
            gridResolution=0.02,
            contourLines=4,
            scale=global_max
        )

        # Generate similarity map
        img = _similarity_map_gen(mol, weights, cmap="bwr", width=600, height=600, **kwargs)

        # Plot image
        ax.imshow(img)
        ax.axis("off")

        # Add colorbar
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

    # Add main title
    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()

    # Save figure if requested
    if output_path:
        fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {output_path}")

    return fig


def compute_ligand_energy_differences(
    protein_components: Dict[str, np.ndarray],
    ligand_components: Dict[str, np.ndarray],
    complex_components: Dict[str, np.ndarray],
    n_protein_atoms: int,
    n_ligand_atoms: int
) -> Dict[str, np.ndarray]:
    """
    Compute per-atom energy differences for ligand atoms.

    Calculates: complex_ligand_part - ligand for each energy component.

    Args:
        protein_components: Per-atom components for protein
        ligand_components: Per-atom components for ligand
        complex_components: Per-atom components for complex
        n_protein_atoms: Number of protein atoms
        n_ligand_atoms: Number of ligand atoms

    Returns:
        dict: Per-atom energy differences for ligand atoms by component
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
            
            # calculate protein energy difference
            protein_part = complex_components[original_comp][:n_protein_atoms]
            protein_alone = protein_components[original_comp]
            protein_diff = protein_part - protein_alone
            protein_differences[mapped_comp] = protein_diff

    # Calculate total as sum of all components
    if ligand_differences:
        ligand_differences['Total'] = sum(ligand_differences.values())
    if protein_differences:
        protein_differences['Total'] = sum(protein_differences.values())

    # convert the arrays to float lists for JSON serialization
    return {comp: values.tolist() for comp, values in ligand_differences.items()}