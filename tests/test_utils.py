"""
Tests for the utils module.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch
from ase import Atoms
from unittest.mock import PropertyMock, patch

from src.utils import (
    write_structure,
    get_supported_formats,
    validate_structure,
    write_opt_structure,
    get_ligand_files
)
from src.molecule_loader import load_ase_structure



class TestReadStructure:
    """Tests for structure reading functionality."""

    
    @pytest.mark.parametrize("format_type", ["xyz", "pdb", "sdf"])
    def test_read_single_structure(self, water_files, format_type):
        """Test reading single structures in different formats."""
        file_path = water_files[format_type]
        print('\n\n\n\n',file_path)
        atoms_list = load_ase_structure(file_path)

        assert isinstance(atoms_list, list)
        assert len(atoms_list) == 1
        atoms = atoms_list[0]
        assert isinstance(atoms, Atoms)
        assert len(atoms) == 3  # Water molecule has 3 atoms
        assert atoms.get_chemical_symbols() == ['O', 'H', 'H']

    
    def test_read_multiple_structures(self, multi_water_file):
        """Test reading multiple structures from SDF file."""
        structures = load_ase_structure(multi_water_file, index=":")
        print('\n\n\n\n',multi_water_file)
        assert isinstance(structures, list)
        assert len(structures) == 2

        for atoms in structures:
            assert isinstance(atoms, Atoms)
            assert len(atoms) == 3
            assert atoms.get_chemical_symbols() == ['O', 'H', 'H']

    
    def test_read_all_index(self, multi_water_file):
        """Test reading all structures with 'all' index."""
        structures = load_ase_structure(multi_water_file, index=":")

        assert isinstance(structures, list)
        assert len(structures) == 2

    @pytest.mark.parametrize("format_type", ["sdf", "xyz", "pdb"])
    def test_read_multiple_structures_various_formats(self, request, format_type):
        """Test reading multiple structures in different formats."""
        # Get the appropriate fixture based on format_type
        if format_type == "sdf":
            file_path = request.getfixturevalue("multi_water_file")
        elif format_type == "xyz":
            file_path = request.getfixturevalue("multi_water_xyz_file")
        elif format_type == "pdb":
            file_path = request.getfixturevalue("multi_water_pdb_file")

        structures = load_ase_structure(file_path, index=":")

        assert isinstance(structures, list)
        assert len(structures) == 2

        for atoms in structures:
            assert isinstance(atoms, Atoms)
            assert len(atoms) == 3
            assert atoms.get_chemical_symbols() == ['O', 'H', 'H']


    def test_read_nonexistent_file(self):
        """Test reading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_ase_structure("nonexistent_file.xyz")

    
    def test_read_invalid_file(self, temp_dir):
        """Test reading invalid file raises error."""
        invalid_file = temp_dir / "invalid.xyz"
        invalid_file.write_text("invalid content")

        with pytest.raises((ValueError, OSError)):
            load_ase_structure(invalid_file)


class TestWriteStructure:
    """Tests for structure writing functionality."""

    
    def test_write_structure_xyz(self, temp_dir, water_files):
        """Test writing structure to XYZ format."""
        atoms = load_ase_structure(water_files['xyz'])[0]
        output_path = temp_dir / "output.xyz"

        result = write_structure(atoms, output_path)

        assert result == str(output_path)
        assert output_path.exists()

        # Verify the written structure can be read back
        read_atoms = load_ase_structure(output_path)[0]
        assert len(read_atoms) == len(atoms)

    
    def test_write_structure_creates_dir(self, temp_dir, water_files):
        """Test that write_structure creates directories if needed."""
        atoms = load_ase_structure(water_files['xyz'])[0]
        output_path = temp_dir / "new_dir" / "output.xyz"

        result = write_structure(atoms, output_path)

        assert result == str(output_path)
        assert output_path.exists()
        assert output_path.parent.exists()

    
    def test_write_structure_auto_format(self, temp_dir, water_files):
        """Test writing structure with auto-detected format from extension."""
        atoms = load_ase_structure(water_files['xyz'])[0]
        output_path = temp_dir / "output.xyz"

        result = write_structure(atoms, output_path)

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
        atoms = load_ase_structure(water_files['xyz'])[0]
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


class TestWriteOptStructure:
    """Tests for write_opt_structure function."""

    def test_write_opt_structure_basic(self, temp_dir, sample_ligand_atoms):
        """Test basic functionality of write_opt_structure."""
        result = write_opt_structure(
            sample_ligand_atoms,
            "opt_ligand",
            "water_opt.xyz",
            temp_dir
        )

        expected_path = temp_dir / "opt_ligand" / "water_opt.xyz"
        assert result == str(expected_path)
        assert expected_path.exists()
        assert expected_path.parent.name == "opt_ligand"

        # Verify the structure can be read back
        read_atoms = load_ase_structure(expected_path)[0]
        assert len(read_atoms) == len(sample_ligand_atoms)

        # Cleanup
        expected_path.unlink()
        expected_path.parent.rmdir()

    def test_write_opt_structure_creates_subdirectories(self, temp_dir, sample_protein_atoms):
        """Test that write_opt_structure creates appropriate subdirectories."""
        result = write_opt_structure(
            sample_protein_atoms,
            "complex",
            "protein_complex.xyz",
            temp_dir
        )

        expected_path = temp_dir / "complex" / "protein_complex.xyz"
        assert result == str(expected_path)
        assert expected_path.exists()
        assert (temp_dir / "complex").is_dir()

        # Cleanup
        expected_path.unlink()
        expected_path.parent.rmdir()

    def test_write_opt_structure_different_types(self, temp_dir, sample_ligand_atoms):
        """Test write_opt_structure with different structure types."""
        structure_types = ["opt_ligand", "complex", "protein"]
        created_paths = []

        for struct_type in structure_types:
            result = write_opt_structure(
                sample_ligand_atoms,
                struct_type,
                f"test_{struct_type}.xyz",
                temp_dir
            )

            expected_path = temp_dir / struct_type / f"test_{struct_type}.xyz"
            assert result == str(expected_path)
            assert expected_path.exists()
            assert (temp_dir / struct_type).is_dir()
            created_paths.append(expected_path)

        # Cleanup all created files and directories
        for path in created_paths:
            path.unlink()
            path.parent.rmdir()


class TestGetLigandFiles:
    """Tests for get_ligand_files function."""

    def test_get_ligand_files_single_file(self, sample_sdf_file):
        """Test get_ligand_files with a single ligand file."""
        result = get_ligand_files(str(sample_sdf_file))

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == str(sample_sdf_file)

    def test_get_ligand_files_multi_structure(self, multi_sdf_file, temp_dir):
        """Test get_ligand_files with multi-molecule SDF file."""
        result = get_ligand_files(str(multi_sdf_file), temp_dir)

        assert isinstance(result, list)
        # Should extract individual ligands if multiple structures found
        if len(result) > 1:
            # Multiple structures were extracted
            for file_path in result:
                assert Path(file_path).exists()
                assert "individual_ligands" in str(file_path)

            # Cleanup extracted files
            import shutil
            extract_dir = temp_dir / "individual_ligands"
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
        else:
            # Single structure, returns original file
            assert result[0] == str(multi_sdf_file)

    def test_get_ligand_files_directory(self, temp_dir, water_files):
        """Test get_ligand_files with a directory containing multiple files."""
        # Create a directory with multiple ligand files
        ligand_dir = temp_dir / "ligands"
        ligand_dir.mkdir()

        # Copy sample files to the directory
        import shutil
        created_files = []

        for i, (fmt, file_path) in enumerate(water_files.items()):
            target = ligand_dir / f"water_{i}.{fmt}"
            shutil.copy(file_path, target)
            created_files.append(target)

        result = get_ligand_files(str(ligand_dir))

        assert isinstance(result, list)
        assert len(result) == len(water_files)

        # Results should be sorted
        result_names = [Path(f).name for f in result]
        expected_names = [f.name for f in created_files]
        for name in expected_names:
            assert name in result_names

        # Cleanup
        shutil.rmtree(ligand_dir)

    def test_get_ligand_files_nonexistent_path(self):
        """Test get_ligand_files with non-existent path."""
        with pytest.raises(FileNotFoundError, match="Ligands input not found"):
            get_ligand_files("/nonexistent/path/ligands.sdf")

    def test_get_ligand_files_empty_directory(self, temp_dir):
        """Test get_ligand_files with empty directory."""
        empty_dir = temp_dir / "empty_ligands"
        empty_dir.mkdir()

        result = get_ligand_files(str(empty_dir))

        assert isinstance(result, list)
        assert len(result) == 0

        # Cleanup
        empty_dir.rmdir()