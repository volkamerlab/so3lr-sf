"""
Molecule Loading Module

This module provides functions for loading molecular structures from various file formats
with proper handling of multi-molecule files.
"""

from typing import Union, List
from pathlib import Path
from tempfile import NamedTemporaryFile
from ase import Atoms
from ase.io import read
from rdkit import Chem
import MDAnalysis as mda
import prolif as plf


def validate_structure(atom_obj: Atoms):
    """Validate that an ASE Atoms object is valid."""
    if not isinstance(atom_obj, Atoms) or len(atom_obj) == 0:
        raise ValueError("Invalid or empty ASE structure.")


def read_xyz(file_path: Path, index: Union[int, str]) -> List[Atoms]:
    """Read XYZ file and return list of Atoms objects."""
    atoms = read(file_path, index=index, format='xyz')
    return [atoms] if isinstance(atoms, Atoms) else atoms

def rdkit_to_ase(mol):
    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = [[p.x, p.y, p.z] for p in [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]]
    return Atoms(symbols=symbols, positions=positions)


def mda_to_ase(u: mda.Universe) -> Atoms:
    """Convert MDAnalysis Universe to ASE Atoms."""
    # Get element symbols, guess if not available
    if hasattr(u.atoms, 'elements'):
        symbols = u.atoms.elements
    else:
        symbols = mda.topology.guessers.guess_types(u.atoms.names)

    positions = u.atoms.positions
    return Atoms(symbols=symbols, positions=positions)

def read_sdf_multi(file_path: Path) -> List[Atoms]:
    """Read multi-molecule SDF file using RDKit and return list of ASE Atoms."""

    supplier = Chem.SDMolSupplier(str(file_path), removeHs=False)
    return [rdkit_to_ase(mol) for mol in supplier if mol is not None]


def read_pdb_multi(file_path: Path) -> List[Atoms]:
    """Read multi-model PDB file using MDAnalysis."""
    try:
        # Load all frames/models from PDB file
        u = mda.Universe(str(file_path))

        molecules = []
        # Iterate through all frames (models in PDB)
        for frame in range(len(u.trajectory)):
            u.trajectory[frame]  # Set to current frame
            molecules.append(mda_to_ase(u))

        return molecules if molecules else [mda_to_ase(u)]

    except Exception as e:
        raise ValueError(f"Could not read PDB file {file_path}: {e}")

def load_ase_structure(file_path: Union[str, Path], index: Union[int, str] = 0) -> List[Atoms]:
    """
    Universal function to load molecular structures to ASE Atoms format.

    Supports: PDB, SDF, XYZ formats for both single and multi-molecule files.
    Always returns a list of ASE Atoms objects.

    Args:
        file_path: Path to structure file
        index: Structure index to read (0 for first, ':' for all, -1 for last)

    Returns:
        List[Atoms]: List of ASE Atoms objects

    Raises:
        ValueError: If file format is not supported or molecule cannot be loaded
    """
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    if index == ":":
        if ext == '.sdf':
            atoms_list = read_sdf_multi(file_path)
        elif ext == '.pdb':
            atoms_list = read_pdb_multi(file_path)
        elif ext == '.xyz':
            atoms_list = read_xyz(file_path, index)
        else:
            raise ValueError(f"Multi-molecule reading not supported for format: {ext}")
    elif ext == '.xyz':
        atoms_list = read_xyz(file_path, index)
    elif ext == '.sdf':
        atoms = read(file_path, index=index, format='sdf')
        atoms_list = [atoms] if isinstance(atoms, Atoms) else atoms
    elif ext == '.pdb':
        atoms = read(file_path, index=index, format='proteindatabank')
        atoms_list = [atoms] if isinstance(atoms, Atoms) else atoms
    else:
        raise ValueError(f"Unsupported file format: {ext}. Only PDB, XYZ, and SDF formats are supported.")

    # Validate that we have structures
    if len(atoms_list) == 0:
        raise ValueError(f"No valid structures found in {file_path}")

    for atom_obj in atoms_list:
        validate_structure(atom_obj)
    return atoms_list


def load_molecule_to_prolif(file_path: Union[str, Path]):
    """
    Universal function to load any molecule (protein or ligand) to ProLIF (RDKit) format.

    Supports: PDB, SDF, XYZ formats for both proteins and ligands

    Args:
        file_path: Path to structure file

    Returns:
        rdkit.Chem.Mol: RDKit molecule object optimized for ProLIF

    Raises:
        ValueError: If file format is not supported or molecule cannot be loaded
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix not in ['.pdb', '.sdf', '.xyz']:
        raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .pdb, .sdf, .xyz")

    # Load RDKit molecule based on format
    if suffix == '.pdb':
        mol = Chem.MolFromPDBFile(str(file_path), removeHs=False)
    elif suffix == '.sdf':
        mol = Chem.MolFromMolFile(str(file_path))
    elif suffix == '.xyz':
        u = mda.Universe(str(file_path))
        # add "elements" category
        elements = mda.topology.guessers.guess_types(u.atoms.names)
        u.add_TopologyAttr("elements", elements)
        mol = plf.Molecule.from_mda(u)
    if mol is None:
        raise ValueError(f"Could not load molecule from {file_path}")

    return mol