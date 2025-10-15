"""
Tests for the molecule_loader module.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from ase import Atoms
import MDAnalysis as mda

from src.molecule_loader import (
    validate_structure,
    read_xyz,
    rdkit_to_ase,
    mda_to_ase,
    read_sdf_multi,
    read_pdb_multi,
    load_ase_structure,
    prepare_mda_universe,
    create_residue_atom_mapping,
    load_molecule_to_prolif
)


class TestMoleculeLoader:
    """Tests for molecule loader functions."""

    def test_validate_structure_valid(self):
        """Test validate_structure with valid Atoms object."""
        atoms = Atoms('H2', positions=[(0, 0, 0), (0, 0, 1)])
        # Should not raise any exception
        validate_structure(atoms)

    def test_validate_structure_empty_structure(self):
        """Test validate_structure with empty structure - MISSING LINE 20."""
        empty_atoms = Atoms()
        with pytest.raises(ValueError, match="Invalid or empty ASE structure"):
            validate_structure(empty_atoms)

    def test_validate_structure_invalid_type(self):
        """Test validate_structure with invalid type - MISSING LINE 20."""
        with pytest.raises(ValueError, match="Invalid or empty ASE structure"):
            validate_structure("not_atoms")

    def test_read_xyz_single_structure(self, water_files):
        """Test reading single structure from XYZ file."""
        result = read_xyz(water_files['xyz'], 0)
        assert len(result) == 1
        assert isinstance(result[0], Atoms)
        assert len(result[0]) == 3  # Water has 3 atoms

    def test_rdkit_to_ase(self):
        """Test conversion from RDKit molecule to ASE Atoms."""
        # Mock RDKit molecule
        mock_mol = Mock()
        mock_conf = Mock()
        mock_atom1 = Mock()
        mock_atom1.GetSymbol.return_value = 'O'
        mock_atom2 = Mock()
        mock_atom2.GetSymbol.return_value = 'H'
        mock_atom3 = Mock()
        mock_atom3.GetSymbol.return_value = 'H'

        mock_mol.GetConformer.return_value = mock_conf
        mock_mol.GetAtoms.return_value = [mock_atom1, mock_atom2, mock_atom3]
        mock_mol.GetNumAtoms.return_value = 3

        # Mock positions
        mock_pos1 = Mock()
        mock_pos1.x, mock_pos1.y, mock_pos1.z = 0.0, 0.0, 0.0
        mock_pos2 = Mock()
        mock_pos2.x, mock_pos2.y, mock_pos2.z = 0.757, 0.586, 0.0
        mock_pos3 = Mock()
        mock_pos3.x, mock_pos3.y, mock_pos3.z = -0.757, 0.586, 0.0

        mock_conf.GetAtomPosition.side_effect = [mock_pos1, mock_pos2, mock_pos3]

        atoms = rdkit_to_ase(mock_mol)
        assert isinstance(atoms, Atoms)
        assert len(atoms) == 3
        assert atoms.get_chemical_symbols() == ['O', 'H', 'H']

    def test_mda_to_ase_with_elements(self):
        """Test conversion from MDAnalysis to ASE with element info."""
        mock_universe = Mock(spec=mda.Universe)
        mock_atoms = Mock()
        mock_atoms.elements = ['O', 'H', 'H']
        mock_atoms.positions = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
        mock_universe.atoms = mock_atoms

        atoms = mda_to_ase(mock_universe)
        assert isinstance(atoms, Atoms)
        assert len(atoms) == 3
        assert atoms.get_chemical_symbols() == ['O', 'H', 'H']

    def test_mda_to_ase_guess_elements(self):
        """Test conversion from MDAnalysis to ASE with element guessing - MISSING LINE 41."""
        mock_universe = Mock(spec=mda.Universe)
        mock_atoms = Mock()
        # No elements attribute, should trigger guessing
        delattr(mock_atoms, 'elements') if hasattr(mock_atoms, 'elements') else None
        mock_atoms.names = ['O1', 'H1', 'H2']
        mock_atoms.positions = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
        mock_universe.atoms = mock_atoms

        with patch('src.molecule_loader.mda.topology.guessers.guess_types') as mock_guess:
            mock_guess.return_value = ['O', 'H', 'H']
            atoms = mda_to_ase(mock_universe)
            assert isinstance(atoms, Atoms)
            assert len(atoms) == 3
            mock_guess.assert_called_once_with(['O1', 'H1', 'H2'])

    def test_read_pdb_multi_exception_handling(self):
        """Test PDB reading with exception handling - MISSING LINES 67-68."""
        nonexistent_file = Path("/nonexistent/file.pdb")

        with pytest.raises(ValueError, match="Could not read PDB file"):
            read_pdb_multi(nonexistent_file)

    def test_read_pdb_multi_with_invalid_pdb_content(self):
        """Test PDB reading with invalid PDB content - MISSING LINES 67-68."""
        with tempfile.NamedTemporaryFile(suffix='.pdb', mode='w', delete=False) as tmp:
            tmp.write("INVALID PDB CONTENT\nNOT A REAL PDB FILE\n")
            tmp_path = Path(tmp.name)

        try:
            with pytest.raises(ValueError, match="Could not read PDB file"):
                read_pdb_multi(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_load_ase_structure_unsupported_format(self):
        """Test loading structure with unsupported format - MISSING LINE 98."""
        with tempfile.NamedTemporaryFile(suffix='.unsupported', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with pytest.raises(ValueError, match="Unsupported file format"):
                load_ase_structure(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_load_ase_structure_nonexistent_file(self):
        """Test loading structure from nonexistent file - MISSING LINE 108."""
        nonexistent_file = Path("/nonexistent/file.xyz")

        with pytest.raises((ValueError, FileNotFoundError)):
            load_ase_structure(nonexistent_file)

    def test_prepare_mda_universe_exception(self):
        """Test prepare_mda_universe with exception - MISSING LINE 134."""
        nonexistent_file = Path("/nonexistent/file.pdb")

        with pytest.raises((ValueError, FileNotFoundError)):
            prepare_mda_universe(nonexistent_file)

    def test_create_residue_atom_mapping_empty(self):
        """Test create_residue_atom_mapping with empty universe - MISSING LINES 160-161."""
        mock_universe = Mock(spec=mda.Universe)
        mock_universe.residues = []

        result = create_residue_atom_mapping(mock_universe)
        assert result == {}

    def test_create_residue_atom_mapping_with_residues(self):
        """Test create_residue_atom_mapping with actual residues - MISSING LINES 160-161."""
        mock_universe = Mock(spec=mda.Universe)

        # Mock atoms for residue 1
        mock_atom1 = Mock()
        mock_atom1.index = 0
        mock_atom2 = Mock()
        mock_atom2.index = 1
        mock_atom3 = Mock()
        mock_atom3.index = 2
        mock_atom4 = Mock()
        mock_atom4.index = 3

        # Mock residue 1
        mock_res1 = Mock()
        mock_res1.resname = 'ALA'
        mock_res1.resid = 1
        mock_res1.segid = 'A'
        mock_res1.atoms = [mock_atom1, mock_atom2, mock_atom3, mock_atom4]

        # Mock atoms for residue 2
        mock_atom5 = Mock()
        mock_atom5.index = 4
        mock_atom6 = Mock()
        mock_atom6.index = 5
        mock_atom7 = Mock()
        mock_atom7.index = 6
        mock_atom8 = Mock()
        mock_atom8.index = 7

        # Mock residue 2
        mock_res2 = Mock()
        mock_res2.resname = 'VAL'
        mock_res2.resid = 2
        mock_res2.segid = 'A'
        mock_res2.atoms = [mock_atom5, mock_atom6, mock_atom7, mock_atom8]

        mock_universe.residues = [mock_res1, mock_res2]

        result = create_residue_atom_mapping(mock_universe)
        expected = {
            'ALA1.A': [0, 1, 2, 3],
            'VAL2.A': [4, 5, 6, 7]
        }
        assert result == expected

    def test_load_molecule_to_prolif_nonexistent_file(self):
        """Test load_molecule_to_prolif with nonexistent file."""
        nonexistent_file = Path("/nonexistent/file.xyz")

        with pytest.raises(ValueError, match="File does not exist"):
            load_molecule_to_prolif(nonexistent_file)

    def test_load_molecule_to_prolif_protein_non_pdb(self):
        """Test load_molecule_to_prolif with protein flag but non-PDB file - MISSING LINES 209-213."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with pytest.raises(ValueError, match="Protein files must be in PDB format"):
                load_molecule_to_prolif(tmp_path, is_protein=True)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_load_molecule_to_prolif_protein_sdf_format(self):
        """Test load_molecule_to_prolif with protein flag but SDF file - MISSING LINES 209-213."""
        with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with pytest.raises(ValueError, match="Protein files must be in PDB format"):
                load_molecule_to_prolif(tmp_path, is_protein=True)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_load_molecule_to_prolif_protein_creation_error(self):
        """Test load_molecule_to_prolif with protein ProLIF creation error - MISSING LINES 218-222."""
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with patch('src.molecule_loader.prepare_mda_universe') as mock_prepare:
                with patch('src.molecule_loader.create_residue_atom_mapping') as mock_mapping:
                    with patch('src.molecule_loader.plf.Molecule.from_mda') as mock_prolif:
                        mock_prepare.return_value = Mock()
                        mock_mapping.return_value = {}
                        mock_prolif.side_effect = Exception("ProLIF error")

                        with pytest.raises(ValueError, match="Could not create ProLIF molecule"):
                            load_molecule_to_prolif(tmp_path, is_protein=True)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_load_molecule_to_prolif_ligand_creation_error(self):
        """Test load_molecule_to_prolif with ligand ProLIF creation error - MISSING LINES 231-232."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with patch('src.molecule_loader.load_ase_structure') as mock_load:
                with patch('src.molecule_loader.plf.Molecule') as mock_prolif:
                    mock_load.return_value = [Atoms('H2', positions=[(0, 0, 0), (0, 0, 1)])]
                    mock_prolif.side_effect = Exception("ProLIF error")

                    with pytest.raises(ValueError, match="Could not load molecule"):
                        load_molecule_to_prolif(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_load_molecule_to_prolif_success_ligand(self, water_files):
        """Test successful ligand loading."""
        result = load_molecule_to_prolif(water_files['xyz'])
        # Should return ProLIF molecule (not testing exact type due to import complexity)
        assert result is not None

    @patch('src.molecule_loader.prepare_mda_universe')
    @patch('src.molecule_loader.create_residue_atom_mapping')
    @patch('src.molecule_loader.plf.Molecule.from_mda')
    def test_load_molecule_to_prolif_success_protein(self, mock_prolif, mock_mapping, mock_prepare):
        """Test successful protein loading."""
        with tempfile.NamedTemporaryFile(suffix='.pdb', delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            mock_universe = Mock()
            mock_prepare.return_value = mock_universe
            mock_mapping.return_value = {'RES1': [0, 1, 2]}
            mock_molecule = Mock()
            mock_prolif.return_value = mock_molecule

            result = load_molecule_to_prolif(tmp_path, is_protein=True)
            assert len(result) == 2  # Should return tuple (molecule, mapping)
            assert result[0] == mock_molecule
            assert result[1] == {'RES1': [0, 1, 2]}
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_load_ase_structure_reading_exception(self):
        """Test load_ase_structure with reading exception during file processing."""
        with tempfile.NamedTemporaryFile(suffix='.xyz', mode='w', delete=False) as tmp:
            tmp.write("INVALID XYZ CONTENT\nNOT VALID\n")
            tmp_path = Path(tmp.name)

        try:
            # This should trigger exception handling in load_ase_structure
            with pytest.raises(ValueError):
                load_ase_structure(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_molecule_loader_edge_cases(self):
        """Test various edge cases to improve coverage."""
        # Test validate_structure with None
        with pytest.raises(ValueError):
            validate_structure(None)

        # Test validate_structure with wrong type
        with pytest.raises(ValueError):
            validate_structure([1, 2, 3])