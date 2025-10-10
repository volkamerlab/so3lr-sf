"""
Protein-Ligand Explainability Module for SO3LR-SF

This module provides advanced explainability functionality for protein-ligand interactions
using ProLIF for interaction fingerprinting and RDKit for visualization of protein-ligand
interaction patterns.
"""

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Union, Optional, Tuple, List
import tempfile
import warnings

# Core dependencies
from rdkit import Chem
from rdkit.Chem import Draw, rdDetermineBonds, rdCoordGen, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry import Point2D

# ProLIF for interaction fingerprinting
import prolif as plf
from prolif.plotting.network import LigNetwork

from .utils import write_structure, load_molecule_to_rdkit


def prepare_protein_ligand_for_prolif(
    protein_path: Union[str, Path],
    ligand_path: Union[str, Path]
) -> Tuple[plf.Molecule, plf.Molecule]:
    """
    Prepare protein and ligand structures for ProLIF analysis.

    Supports: PDB, SDF, XYZ formats for both proteins and ligands

    Args:
        protein_path: Path to protein structure file
        ligand_path: Path to ligand structure file

    Returns:
        tuple: (protein_mol, ligand_mol) prepared for ProLIF
    """
    protein_path = Path(protein_path)
    ligand_path = Path(ligand_path)

    # Load molecules using universal function
    rdkit_protein = load_molecule_to_rdkit(protein_path)
    rdkit_ligand = load_molecule_to_rdkit(ligand_path)

    # For XYZ files, add residue information for ProLIF
    if protein_path.suffix.lower() == '.xyz':
        rdkit_protein = _add_residue_info_to_mol(rdkit_protein, "PROT", "A")

    if ligand_path.suffix.lower() == '.xyz':
        rdkit_ligand = _add_residue_info_to_mol(rdkit_ligand, "LIG", "A")

    # Convert to ProLIF molecules
    protein_mol = plf.Molecule.from_rdkit(rdkit_protein)
    ligand_mol = plf.Molecule.from_rdkit(rdkit_ligand)

    return protein_mol, ligand_mol


def _add_residue_info_to_mol(mol: Chem.Mol, res_name: str = "UNK", chain_id: str = "A") -> Chem.Mol:
    """
    Add residue information to RDKit molecule for ProLIF compatibility.

    Args:
        mol: RDKit molecule
        res_name: Residue name (default: "UNK")
        chain_id: Chain ID (default: "A")

    Returns:
        Modified RDKit molecule with residue info
    """
    from rdkit.Chem import AtomPDBResidueInfo

    # Create a writable copy
    mol = Chem.RWMol(mol)

    # Add residue info to each atom
    for atom in mol.GetAtoms():
        info = AtomPDBResidueInfo()
        info.SetResidueName(res_name)
        info.SetResidueNumber(1)
        info.SetChainId(chain_id)
        info.SetName(f"{atom.GetSymbol()}{atom.GetIdx()}")
        info.SetIsHeteroAtom(res_name not in ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                                               'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                                               'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                                               'SER', 'THR', 'TRP', 'TYR', 'VAL'])
        atom.SetMonomerInfo(info)

    return mol.GetMol()


def compute_protein_ligand_interactions(
    protein_mol: plf.Molecule,
    ligand_mol: plf.Molecule,
    interactions: Optional[List[str]] = None
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
    # Default interaction types if not specified
    if interactions is None:
        interactions = [
            "Hydrophobic", "HBDonor", "HBAcceptor", "PiStacking",
            "Anionic", "Cationic", "CationPi", "PiCation", "VdWContact"
        ]

    # Create fingerprint generator
    fp = plf.Fingerprint(interactions)

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
              [
                  {
                      'ligand_residue': 'UNL1',
                      'protein_residue': 'MET1211.A',
                      'interaction_type': 'Hydrophobic',
                      'ligand_atoms': [7],
                      'protein_atoms': [15],
                      'distance': 3.5,
                      ...
                  },
                  ...
              ]
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
                        mapping['distance'] = round(metadata['distance'], 2)
                    if 'DHA_angle' in metadata:
                        mapping['DHA_angle'] = round(metadata['DHA_angle'], 2)

                    mappings.append(mapping)

    return mappings


def custom_interactions_to_mappings(
    interactions: List[Tuple[str, Tuple[int, ...], str, Optional[float]]],
    ligand_residue: str = "UNL1"
) -> List[Dict[str, Any]]:
    """
    Convert custom interaction format to standard atom mapping format.

    Args:
        interactions: List of (residue, ligand_atoms, interaction_type, distance)
                     Example: [('LYS 89', (3,), 'hydrophobic', 3.5),
                               ('ASP 86', (4,), 'hbond', 2.8)]
        ligand_residue: Name of the ligand residue (default: "UNL1")

    Returns:
        list: Standardized atom mappings compatible with visualization functions
    """
    mappings = []

    for interaction in interactions:
        if len(interaction) == 3:
            residue, ligand_atoms, interaction_type = interaction
            distance = None
        else:
            residue, ligand_atoms, interaction_type, distance = interaction

        mapping = {
            'frame': 0,
            'ligand_residue': ligand_residue,
            'protein_residue': residue,
            'interaction_type': interaction_type,
            'ligand_atoms': list(ligand_atoms),
            'protein_atoms': []  # Not provided in custom format
        }

        if distance is not None:
            mapping['distance'] = round(distance, 2)

        mappings.append(mapping)

    return mappings


def visualize_interactions(
    ligand_path: Union[str, Path],
    atom_mappings: List[Dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    size: Tuple[int, int] = (800, 600)
) -> plt.Figure:
    """
    Visualize protein-ligand interactions with atom-level highlighting and residue labels.

    Args:
        ligand_path: Path to ligand structure file
        atom_mappings: List of atom-level interaction mappings (from get_atom_mappings
                      or custom_interactions_to_mappings)
        output_path: Optional path to save the figure
        title: Optional title for the figure
        size: Size of the drawing (width, height) in pixels

    Returns:
        plt.Figure: Generated matplotlib figure
    """
    ligand_path = Path(ligand_path)
    rdkit_mol = load_molecule_to_rdkit(ligand_path)

    if rdkit_mol is None:
        raise ValueError(f"Could not load ligand from {ligand_path}")

    # Convert to SMILES and back to fix coordinate issues
    smiles = Chem.MolToSmiles(rdkit_mol)
    rdkit_mol = Chem.MolFromSmiles(smiles)

    # Create writable copy for adding pseudo-atoms
    lig_with_interactions = Chem.RWMol(rdkit_mol)

    # Build atom-to-interaction mapping and color
    atom_colors = {}
    pts = []
    highlight_bonds = []
    highlight_bond_colors = {}
    seen_bonds = set()
    highlight_atom_radii = {}

    for mapping in atom_mappings:
        # Create residue atom
        res_atom = Chem.Atom(0)
        res_atom.SetProp('atomLabel', mapping['protein_residue'])
        aid = lig_with_interactions.AddAtom(res_atom)
        pts.append(aid)
        atom_colors[aid] = (1, .2, 1, .3)  # Light magenta for residue pseudo-atom

        # Bond color
        bond_color = _get_interaction_color([mapping['interaction_type']])
        highlight_atom_radii[aid] = 0.5

        for atom_idx in mapping.get("ligand_atoms", []):
            lig_with_interactions.AddBond(aid, atom_idx, Chem.BondType.ZERO)
            bond = lig_with_interactions.GetBondBetweenAtoms(aid, atom_idx)
            if bond is None:
                continue
            bidx = bond.GetIdx()
            if bidx in seen_bonds:
                continue
            seen_bonds.add(bidx)
            highlight_bonds.append(bidx)
            highlight_bond_colors[bidx] = bond_color

    # Draw molecule using Cairo (PNG) drawer
    d2d = Draw.MolDraw2DCairo(size[0], size[1])
    d2d.drawOptions().circleAtoms = True
    d2d.drawOptions().fillHighlights = True
    d2d.drawOptions().continuousHighlight = False
    d2d.drawOptions().highlightRadius = 0.5

    # Draw molecule with highlighting
    d2d.DrawMolecule(
        lig_with_interactions,
        legend=title or "Protein-Ligand Interactions",
        highlightAtoms=pts,
        highlightBonds=highlight_bonds,
        highlightBondColors=highlight_bond_colors,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=highlight_atom_radii
    )
    d2d.FinishDrawing()
    raw = d2d.GetDrawingText()

    # Build atom_interactions dict for legend
    atom_interactions = {}
    for mapping in atom_mappings:
        for atom_idx in mapping.get('ligand_atoms', []):
            if atom_idx not in atom_interactions:
                atom_interactions[atom_idx] = []
            atom_interactions[atom_idx].append(mapping['interaction_type'])

    # Convert PNG bytes to PIL image then to matplotlib
    if isinstance(raw, bytes):
        img = Image.open(io.BytesIO(raw)).convert("RGBA")
        fig, ax = plt.subplots(1, 1, figsize=(12, 9))
        ax.imshow(img)
        ax.axis("off")

        _add_interaction_legend(fig, atom_interactions)
        _add_interaction_summary(fig, atom_mappings)

        plt.tight_layout()

        if output_path:
            fig.savefig(str(output_path), dpi=300, bbox_inches='tight')

        return fig
    else:
        # Handle SVG case if needed
        raise ValueError("SVG output not supported in this version")


def create_interaction_network(
    fp: plf.Fingerprint,
    ligand_mol: plf.Molecule,
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Create network visualization of protein-ligand interactions using ProLIF's network plotting.

    Note: This function requires specific DataFrame format that may not be available
    with all ProLIF fingerprint configurations. Returns None if network cannot be created.

    Args:
        fp: ProLIF Fingerprint object with interaction data
        ligand_mol: ProLIF ligand molecule object
        output_path: Optional path to save the figure
        title: Optional title for the figure

    Returns:
        plt.Figure or None: Generated matplotlib figure, or None if network creation fails
    """
    try:
        # Try to create network plot - this requires specific DataFrame format
        df = fp.to_dataframe()

        # Check if the DataFrame has the expected structure for LigNetwork
        if 'atoms' not in df.index.names:
            print("Warning: DataFrame format not compatible with LigNetwork. Skipping network visualization.")
            return None

        net = LigNetwork(df=df, lig_mol=ligand_mol)

        # Generate the network visualization
        fig = net.plot(
            figsize=(12, 8),
            threshold=0.3,  # Only show interactions above 30% frequency
            rotation=0
        )

        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold')

        # Save figure if requested
        if output_path:
            fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
            print(f"Interaction network saved to: {output_path}")

        return fig

    except (KeyError, ValueError, AttributeError) as e:
        print(f"Warning: Could not create network visualization: {e}")
        print("This is expected with certain ProLIF configurations. Skipping network plot.")
        return None


def analyze_interaction_patterns(atom_mappings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze interaction patterns from atom-level mappings.

    Args:
        atom_mappings: List of atom-level interaction mappings from get_atom_mappings()

    Returns:
        dict: Analysis results including frequencies, residue involvement, etc.
    """
    analysis = {}

    # Count interactions by type
    interaction_counts = {}
    residue_involvement = {}

    for mapping in atom_mappings:
        interaction_type = mapping['interaction_type']
        protein_residue = mapping['protein_residue']

        # Count interaction types
        interaction_counts[interaction_type] = interaction_counts.get(interaction_type, 0) + 1

        # Count residue involvement
        residue_involvement[protein_residue] = residue_involvement.get(protein_residue, 0) + 1

    total_interactions = len(atom_mappings)
    analysis['total_interactions'] = total_interactions
    analysis['interaction_frequencies'] = interaction_counts

    if total_interactions > 0:
        analysis['interaction_percentages'] = {
            interaction: float(count / total_interactions * 100)
            for interaction, count in interaction_counts.items()
        }
    else:
        analysis['interaction_percentages'] = {}

    analysis['residue_involvement'] = dict(sorted(
        residue_involvement.items(), key=lambda x: x[1], reverse=True
    ))

    analysis['interaction_diversity'] = len(interaction_counts)

    return analysis


def generate_comprehensive_interaction_report(
    protein_path: Union[str, Path],
    ligand_path: Union[str, Path],
    output_dir: Union[str, Path],
    ligand_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a comprehensive interaction analysis report.

    Args:
        protein_path: Path to protein structure file (PDB or XYZ)
        ligand_path: Path to ligand structure file (SDF or XYZ)
        output_dir: Directory to save output files
        ligand_name: Optional name for the ligand

    Returns:
        dict: Comprehensive analysis results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if ligand_name is None:
        ligand_name = Path(ligand_path).stem

    # Prepare molecules
    protein_mol, ligand_mol = prepare_protein_ligand_for_prolif(protein_path, ligand_path)

    # Compute interactions - returns fingerprint object
    fp = compute_protein_ligand_interactions(protein_mol, ligand_mol)

    # Get atom-level mappings
    atom_mappings = get_atom_mappings(fp)

    # Analyze patterns
    analysis = analyze_interaction_patterns(atom_mappings)

    # Generate visualizations
    # 2D interaction visualization
    interaction_2d_path = output_dir / f"{ligand_name}_interactions_2d.png"
    fig_2d = visualize_interactions(
        ligand_path, atom_mappings,
        output_path=interaction_2d_path,
        title=f"Protein-Ligand Interactions: {ligand_name}"
    )
    plt.close(fig_2d)

    # Network visualization
    network_path = output_dir / f"{ligand_name}_interaction_network.png"
    fig_network = create_interaction_network(
        fp,
        ligand_mol,
        output_path=network_path,
        title=f"Interaction Network: {ligand_name}"
    )
    if fig_network is not None:
        plt.close(fig_network)

    # Save atom mapping data
    atom_mapping_csv_path = output_dir / f"{ligand_name}_atom_mappings.csv"
    pd.DataFrame(atom_mappings).to_csv(atom_mapping_csv_path, index=False)

    # Compile results
    results = {
        'ligand_name': ligand_name,
        'analysis': analysis,
        'atom_mappings': atom_mappings,
        'files': {
            'interaction_2d': str(interaction_2d_path),
            'interaction_network': str(network_path),
            'atom_mappings': str(atom_mapping_csv_path)
        }
    }

    return results


def _get_interaction_color(interactions: List[str]) -> Tuple[float, float, float]:
    """
    Get color for atom based on interaction types.

    Color scheme:
    - Blue: H-bonds (HBDonor, HBAcceptor)
    - Dark Blue: Halogen bonds (XBDonor, XBAcceptor)
    - Green: Hydrophobic and VdW contacts
    - Yellow: Pi interactions (PiStacking, PiCation, CationPi, EdgeToFace, FaceToFace, BasePiStacking)
    - Cyan: Ionic (Anionic, Cationic)
    - Purple: Metal interactions (MetalDonor, MetalAcceptor)
    - Gray: Distance/Angle measurements

    Args:
        interactions: List of interaction types for an atom

    Returns:
        tuple: RGB color as (R, G, B) with values 0-1
    """
    # Normalize interaction types to lowercase for comparison
    interactions_lower = [i.lower() for i in interactions]

    # Priority order: H-bond > Halogen > Hydrophobic/VdW > Pi > Ionic > Metal > Distance/Angle
    if any(x in interactions_lower for x in ['hbdonor', 'hbacceptor', 'hbond', 'hydrophilic', 'h-bond']):
        return (0.2, 0.4, 0.9)  # Blue
    elif any(x in interactions_lower for x in ['xbdonor', 'xbacceptor', 'halogen']):
        return (0.1, 0.2, 0.5)  # Dark Blue
    elif any(x in interactions_lower for x in ['hydrophobic', 'vdwcontact', 'vdw', 'vanderwaals']):
        return (0.2, 0.8, 0.2)  # Green
    elif any(x in interactions_lower for x in ['pistacking', 'pication', 'cationpi', 'edgetoface', 'facetoface', 'basepistacking', 'pi-stacking', 'pi']):
        return (0.9, 0.9, 0.2)  # Yellow
    elif any(x in interactions_lower for x in ['anionic', 'cationic', 'saltbridge', 'ionic', 'salt-bridge']):
        return (0.3, 0.8, 0.8)  # Cyan
    elif any(x in interactions_lower for x in ['metaldonor', 'metalacceptor', 'metal']):
        return (0.8, 0.2, 0.8)  # Purple/Magenta
    elif any(x in interactions_lower for x in ['distance', 'singleangle', 'doubleangle', 'angle']):
        return (0.6, 0.6, 0.6)  # Gray
    else:
        return (0.5, 0.5, 0.5)  # Default gray


def _add_interaction_legend(fig: plt.Figure, atom_interactions: Dict[int, List[str]]) -> plt.Figure:
    """Add legend for interaction types to the figure, only showing colors that exist in the data."""
    # Collect all unique interaction types from atom_interactions
    all_interactions = set()
    for interactions_list in atom_interactions.values():
        all_interactions.update([i.lower() for i in interactions_list])

    # Define all possible color mappings
    color_map = {
        'H-bond': ((0.2, 0.4, 0.9), ['hbdonor', 'hbacceptor', 'hbond', 'hydrophilic', 'h-bond']),
        'Halogen bond': ((0.1, 0.2, 0.5), ['xbdonor', 'xbacceptor', 'halogen']),
        'Hydrophobic/VdW': ((0.2, 0.8, 0.2), ['hydrophobic', 'vdwcontact', 'vdw', 'vanderwaals']),
        'π-interaction': ((0.9, 0.9, 0.2), ['pistacking', 'pication', 'cationpi', 'edgetoface', 'facetoface', 'basepistacking', 'pi-stacking', 'pi']),
        'Ionic': ((0.3, 0.8, 0.8), ['anionic', 'cationic', 'saltbridge', 'ionic', 'salt-bridge']),
        'Metal': ((0.8, 0.2, 0.8), ['metaldonor', 'metalacceptor', 'metal']),
        'Distance/Angle': ((0.6, 0.6, 0.6), ['distance', 'singleangle', 'doubleangle', 'angle'])
    }

    # Build legend elements only for interactions that exist in the data
    legend_elements = []
    for label, (color, keywords) in color_map.items():
        if any(keyword in all_interactions for keyword in keywords):
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                          markersize=10, label=label)
            )

    if legend_elements:
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    return fig


def _add_interaction_summary(fig: plt.Figure, atom_mappings: List[Dict[str, Any]]) -> None:
    """Add interaction summary text to the figure."""
    # Count total interactions
    total_interactions = len(atom_mappings)
    interaction_types = len(set(m['interaction_type'] for m in atom_mappings))

    summary_text = f"Total Interactions: {total_interactions}\nInteraction Types: {interaction_types}"

    fig.text(0.02, 0.98, summary_text, transform=fig.transFigure,
             verticalalignment='top', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))