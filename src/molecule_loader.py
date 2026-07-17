"""
Molecule Loading Module

This module provides functions for loading molecular structures from various file formats
with proper handling of multi-molecule files.
"""

from typing import Union, List, Dict, Tuple, Optional
from pathlib import Path
from ase import Atoms
from ase.io import read
from rdkit import Chem
import MDAnalysis as mda
import prolif as plf
import logging

logger = logging.getLogger(__name__)


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


def create_residue_atom_mapping(universe: mda.Universe) -> Dict[str, List[int]]:
    """
    Create mapping of residue identifiers to their atom indices.

    Args:
        universe: MDAnalysis Universe object

    Returns:
        Dict mapping "resname+resid.segid" to list of atom indices

    Example:
        >>> u = mda.Universe("protein.pdb")
        >>> mapping = create_residue_atom_mapping(u)
        >>> # mapping["ALA1.A"] = [0, 1, 2, 3, 4]
    """
    return {
        f"{residue.resname}{residue.resid}.{residue.segid}": [int(atom.index) for atom in residue.atoms]
        for residue in universe.residues
    }




def prepare_mda_universe(file_path: Path) -> mda.Universe:
    """
    Helper function to create and prepare MDAnalysis Universe with elements.

    Args:
        file_path: Path to structure file

    Returns:
        Prepared MDAnalysis Universe with elements topology attribute

    Raises:
        ValueError: If Universe cannot be created
    """
    try:
        universe = mda.Universe(str(file_path))
        elements = mda.topology.guessers.guess_types(universe.atoms.names)
        universe.add_TopologyAttr("elements", elements)
        return universe
    except Exception as e:
        raise ValueError(f"Could not create MDAnalysis Universe from {file_path}: {e}")


def load_molecule_to_prolif(
    file_path: Union[str, Path],
    is_protein: bool = False,
) -> Union[plf.Molecule, Tuple[plf.Molecule, Dict[str, List[int]]]]:
    """
    Universal function to load molecules to ProLIF format with optional residue mapping.

    Supports PDB, SDF, XYZ formats for both proteins and ligands.
    For proteins, requires PDB format to extract residue information.

    Args:
        file_path: Path to structure file
        is_protein: If True, returns residue mapping (requires PDB format)

    Returns:
        If is_protein=False: prolif.Molecule object
        If is_protein=True: Tuple of (prolif.Molecule, residue_atom_mapping dict)

    Raises:
        ValueError: If file format is not supported, molecule cannot be loaded,
                   or protein flag is True but file is not PDB format

    Example:
        >>> # Load ligand
        >>> ligand = load_molecule_to_prolif("ligand.sdf")
        >>>
        >>> # Load protein with residue mapping
        >>> protein, mapping = load_molecule_to_prolif("protein.pdb", is_protein=True)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise ValueError(f"File does not exist: {file_path}")

    suffix = file_path.suffix.lower()
    supported_formats = {'.pdb', '.sdf', '.xyz'}

    if suffix not in supported_formats:
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: {', '.join(sorted(supported_formats))}"
        )

    # Handle protein case (requires PDB for residue information)
    if is_protein:
        if suffix != '.pdb':
            raise ValueError(
                f"Protein files must be in PDB format to extract residue information. "
                f"Got: {suffix}. Use is_protein=False for non-PDB protein files."
            )

        universe = prepare_mda_universe(file_path)
        residue_mapping = create_residue_atom_mapping(universe)

        try:
            molecule = plf.Molecule.from_mda(universe)
            return molecule, residue_mapping
        except Exception as e:
            raise ValueError(f"Could not create ProLIF molecule from {file_path}: {e}")

    # Handle ligand/non-protein cases
    try:
        if suffix == '.pdb':
            # Try RDKit PDB parser first (better for small molecules)
            mol = Chem.MolFromPDBFile(str(file_path), removeHs=False)
            if mol is None:
                # Fallback to MDAnalysis for complex PDB files
                universe = prepare_mda_universe(file_path)
                return plf.Molecule.from_mda(universe)
            return plf.Molecule(mol)

        elif suffix == '.sdf':
            mol = Chem.MolFromMolFile(str(file_path), removeHs=False)
            if mol is None:
                raise ValueError(f"RDKit could not parse SDF file: {file_path}")
            return plf.Molecule(mol)

        elif suffix == '.xyz':
            # XYZ files need MDAnalysis for proper handling
            universe = prepare_mda_universe(file_path)
            return plf.Molecule.from_mda(universe)

    except Exception as e:
        raise ValueError(f"Could not load molecule from {file_path}: {e}")


def extract_ligands(
    multi_structure_file: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    naming_prefix: str = "ligand"
) -> List[str]:
    """
    Extract individual ligands from multi-structure files (SDF, XYZ, PDB with multiple structures).

    This function reads a file containing multiple structures and saves each one
    as a separate XYZ file for individual processing.

    Args:
        multi_structure_file: Path to file containing multiple structures
        output_dir: Directory to save individual ligand files
        naming_prefix: Prefix for output filenames

    Returns:
        List[str]: Paths to individual ligand XYZ files

    Example:
        >>> # Extract ligands from multi-molecule SDF
        >>> ligand_files = extract_ligands("ligands.sdf", output_dir="individual_ligands")
        >>> for lig_file in ligand_files:
        ...     energy = energy_calc_fn(lig_file)
        ...     print(f"{lig_file}: {energy:.3f} eV")
    """
    from .utils import write_structure  # Import here to avoid circular imports

    multi_structure_file = Path(multi_structure_file)

    if output_dir is None:
        output_dir = multi_structure_file.parent / f"{multi_structure_file.stem}_individual"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Read all structures
    structures = load_ase_structure(multi_structure_file, index=":")

    output_files = []

    for i, atoms in enumerate(structures):

        # Generate filename
        output_filename = f"{naming_prefix}_{i+1:03d}.xyz"
        output_path = output_dir / output_filename

        # Save structure
        output_file = write_structure(atoms, output_path)
        output_files.append(output_file)

    logger.info(f"Extracted {len(output_files)} structures to {output_dir}")

    return output_files