"""
Tests for the utils module.
"""

import pytest
import numpy as np
from unittest.mock import patch
from ase import Atoms
from unittest.mock import PropertyMock, patch

from src.utils import (
    read_structure,
    write_structure,
    get_supported_formats,
    validate_structure
)



class TestReadStructure:
    """Tests for structure reading functionality."""

    
    @pytest.mark.parametrize("format_type", ["xyz", "pdb", "sdf"])
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
        print('\n\n\n\n',multi_water_file)
        assert isinstance(structures, list)
        assert len(structures) == 2

        for atoms in structures:
            assert isinstance(atoms, Atoms)
            assert len(atoms) == 3
            assert atoms.get_chemical_symbols() == ['O', 'H', 'H']

    
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
        output_path = temp_dir / "output.xyz"

        result = write_structure(atoms, output_path, format="xyz")

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
        # Create an atoms object and manually set positions to None
        atoms = Atoms(['H'])
        # Monkey patch the positions property to return None
        with patch.object(Atoms, 'positions', new_callable=PropertyMock) as mock_pos:
            mock_pos.side_effect = AttributeError("No positions")
            with pytest.raises(ValueError, match="Structure has no atomic positions"):
                validate_structure(atoms)

    
    # def test_validate_structure_position_mismatch(self):
    #     """Test validation fails when positions don't match atom count."""

    #     atoms = Atoms(['H', 'H'])
    #     with patch.object(Atoms, 'positions', new_callable=PropertyMock) as mock_pos:
    #         mock_pos.return_value = np.array([[0, 0, 0]])  # Mismatch: only 1 position
    #         with pytest.raises(ValueError, match="Mismatch between atom count and position count"):
    #             validate_structure(atoms)

    
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

    
    def test_supported_formats_content(self):
        """Test specific content of supported formats."""
        formats = get_supported_formats()

        expected_formats = ['.xyz', '.pdb', '.sdf']
        for fmt in expected_formats:
            assert fmt in formats