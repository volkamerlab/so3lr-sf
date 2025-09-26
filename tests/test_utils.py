"""
Tests for the utils module.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, Mock
from ase import Atoms

from src.utils import (
    find_so3lr_params,
    read_structure,
    write_structure,
    get_supported_formats,
    validate_structure
)


class TestFindSo3lrParams:
    """Tests for SO3LR model parameter detection."""

    
    def test_find_existing_params_relative(self, temp_dir):
        """Test finding existing SO3LR parameters relative to project."""
        # Create fake project structure
        params_dir = temp_dir / "so3lr" / "so3lr" / "params"
        params_dir.mkdir(parents=True)

        # Mock the current file path to be in src/ directory
        mock_file_path = temp_dir / "src" / "utils.py"
        mock_file_path.parent.mkdir(parents=True)
        mock_file_path.touch()

        with patch('src.utils.Path.__file__', str(mock_file_path)):
            result = find_so3lr_params()
            assert result == str(params_dir)

    
    def test_find_params_absolute_path(self):
        """Test finding parameters using absolute path."""
        absolute_path = Path("/home/hamza/github/so3lr-sf/so3lr/so3lr/params")

        # Create a mock that returns True for is_dir() only for our specific path
        def mock_is_dir(self):
            return str(self) == str(absolute_path)

        with patch.object(Path, 'is_dir', mock_is_dir):
            result = find_so3lr_params()
            assert result == str(absolute_path)

    
    def test_find_params_not_found(self, temp_dir):
        """Test when SO3LR parameters are not found."""
        # Mock the current file path to be in a temp directory with no params
        mock_file_path = temp_dir / "src" / "utils.py"
        mock_file_path.parent.mkdir(parents=True)
        mock_file_path.touch()

        with patch('src.utils.Path.__file__', str(mock_file_path)):
            with patch.object(Path, 'is_dir', return_value=False):
                result = find_so3lr_params()
                assert result is None


class TestReadStructure:
    """Tests for structure reading functionality."""

    
    @pytest.mark.parametrize("format_type", ["xyz", "pdb", "sdf", "mol2"])
    def test_read_single_structure(self, water_files, format_type):
        """Test reading single structures in different formats."""
        file_path = water_files[format_type]
        atoms = read_structure(file_path)

        assert isinstance(atoms, Atoms)
        assert len(atoms) == 3  # Water molecule has 3 atoms
        assert atoms.get_chemical_symbols() == ['O', 'H', 'H']

    
    def test_read_multiple_structures(self, multi_water_file):
        """Test reading multiple structures from SDF file."""
        structures = read_structure(multi_water_file, index=":")

        assert isinstance(structures, list)
        assert len(structures) == 2

        for atoms in structures:
            assert isinstance(atoms, Atoms)
            assert len(atoms) == 3
            assert atoms.get_chemical_symbols() == ['O', 'H', 'H']

    
    def test_read_specific_index(self, multi_water_file):
        """Test reading specific structure index."""
        atoms = read_structure(multi_water_file, index=1)

        assert isinstance(atoms, Atoms)
        assert len(atoms) == 3

    
    def test_read_last_structure(self, multi_water_file):
        """Test reading last structure."""
        atoms = read_structure(multi_water_file, index=-1)

        assert isinstance(atoms, Atoms)
        assert len(atoms) == 3

    
    def test_read_all_index(self, multi_water_file):
        """Test reading all structures with 'all' index."""
        structures = read_structure(multi_water_file, index="all")

        assert isinstance(structures, list)
        assert len(structures) == 2

    
    def test_read_nonexistent_file(self):
        """Test reading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            read_structure("nonexistent_file.xyz")

    
    def test_read_invalid_file(self, temp_dir):
        """Test reading invalid file raises error."""
        invalid_file = temp_dir / "invalid.xyz"
        invalid_file.write_text("invalid content")

        with pytest.raises(ValueError, match="Could not read structure"):
            read_structure(invalid_file)


class TestWriteStructure:
    """Tests for structure writing functionality."""

    
    def test_write_structure_xyz(self, temp_dir, water_files):
        """Test writing structure to XYZ format."""
        atoms = read_structure(water_files['xyz'])
        output_path = temp_dir / "output.xyz"

        result = write_structure(atoms, output_path)

        assert result == str(output_path)
        assert output_path.exists()

        # Verify the written structure can be read back
        read_atoms = read_structure(output_path)
        assert len(read_atoms) == len(atoms)

    
    def test_write_structure_creates_dir(self, temp_dir, water_files):
        """Test that write_structure creates directories if needed."""
        atoms = read_structure(water_files['xyz'])
        output_path = temp_dir / "new_dir" / "output.xyz"

        result = write_structure(atoms, output_path)

        assert result == str(output_path)
        assert output_path.exists()
        assert output_path.parent.exists()

    
    def test_write_structure_with_format(self, temp_dir, water_files):
        """Test writing structure with explicit format."""
        atoms = read_structure(water_files['xyz'])
        output_path = temp_dir / "output.pdb"

        result = write_structure(atoms, output_path, format="pdb")

        assert result == str(output_path)
        assert output_path.exists()

    
    def test_write_structure_error_handling(self, temp_dir):
        """Test error handling in write_structure."""
        atoms = Atoms(['H'], positions=[[0, 0, 0]])
        # Try to write to a path that would cause an error
        invalid_path = temp_dir / "invalid" / "path" / "that" / "is" / "too" / "deep" / "output.xyz"

        # Mock the ASE write function to raise an exception
        with patch('src.utils.write', side_effect=Exception("Mock write error")):
            with pytest.raises(RuntimeError, match="Failed to write structure"):
                write_structure(atoms, invalid_path)


class TestValidateStructure:
    """Tests for structure validation."""

    
    def test_validate_valid_structure(self, water_files):
        """Test validation of valid structure."""
        atoms = read_structure(water_files['xyz'])
        result = validate_structure(atoms)
        assert result is True

    
    def test_validate_empty_structure(self):
        """Test validation fails for empty structure."""
        atoms = Atoms()

        with pytest.raises(ValueError, match="Structure contains no atoms"):
            validate_structure(atoms)

    
    def test_validate_structure_no_positions(self):
        """Test validation fails for structure without positions."""
        atoms = Atoms(['H'])
        atoms.positions = None

        with pytest.raises(ValueError, match="Structure has no atomic positions"):
            validate_structure(atoms)

    
    def test_validate_structure_position_mismatch(self):
        """Test validation fails when positions don't match atom count."""
        atoms = Atoms(['H', 'H'])
        atoms.positions = np.array([[0, 0, 0]])  # Only 1 position for 2 atoms

        with pytest.raises(ValueError, match="Number of positions doesn't match"):
            validate_structure(atoms)

    
    def test_validate_structure_nan_coordinates(self):
        """Test validation fails for NaN coordinates."""
        atoms = Atoms(['H'], positions=[[np.nan, 0, 0]])

        with pytest.raises(ValueError, match="invalid.*NaN.*coordinates"):
            validate_structure(atoms)

    
    def test_validate_structure_infinite_coordinates(self):
        """Test validation fails for infinite coordinates."""
        atoms = Atoms(['H'], positions=[[np.inf, 0, 0]])

        with pytest.raises(ValueError, match="invalid.*infinite.*coordinates"):
            validate_structure(atoms)


class TestGetSupportedFormats:
    """Tests for supported file formats."""

    
    def test_get_supported_formats(self):
        """Test that supported formats are returned."""
        formats = get_supported_formats()

        assert isinstance(formats, list)
        assert len(formats) > 0
        assert '.xyz' in formats
        assert '.pdb' in formats
        assert '.sdf' in formats
        assert '.mol2' in formats

    
    def test_supported_formats_content(self):
        """Test specific content of supported formats."""
        formats = get_supported_formats()

        expected_formats = ['.xyz', '.pdb', '.sdf', '.mol', '.mol2', '.cif', '.traj', '.vasp', '.poscar']
        for fmt in expected_formats:
            assert fmt in formats