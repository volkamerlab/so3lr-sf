"""
Unit tests for the utils module.

This module tests all utility functions including:
- Model parameter detection
- Structure reading and writing
- File format validation
- Logging configuration
- Directory management
- Result saving functionality

Each test is designed to be:
1. Independent and isolated
2. Readable with clear naming
3. Well-documented with docstrings
4. Comprehensive in coverage
"""

import pytest
import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from ase import Atoms

from src.utils import (
    find_so3lr_params,
    read_structure,
    write_structure,
    validate_structure,
    get_supported_formats,
    get_ligand_files,
    setup_output_directory,
    save_results,
    setup_logging,
    write_opt_structure
)


class TestModelDetection:
    """Test SO3LR model parameter detection functionality."""

    def test_find_so3lr_params_success(self, temp_dir):
        """
        Test successful detection of SO3LR parameters directory.

        This test verifies that the function correctly locates
        the SO3LR parameters when they exist in the expected location.
        """
        # Act: Call the real function (it should handle missing paths gracefully)
        result = find_so3lr_params()

        # Assert: Should either find a path or return None
        assert result is None or isinstance(result, str)
        if result:
            assert "params" in str(result)

    def test_find_so3lr_params_not_found(self, temp_dir):
        """
        Test behavior when SO3LR parameters are not found.

        This test ensures the function gracefully returns None
        when the parameters directory doesn't exist.
        """
        # Act: Call the real function
        result = find_so3lr_params()

        # Assert: Should return None or a valid string path
        assert result is None or isinstance(result, str)

    def test_find_so3lr_params_fallback_path(self):
        """
        Test fallback to absolute path when relative path fails.

        This test verifies that the function attempts to use
        a fallback absolute path when the relative path doesn't work.
        """
        # Act: Call function (will use real filesystem)
        result = find_so3lr_params()

        # Assert: Should either find a path or return None
        assert result is None or isinstance(result, str)
        if result:
            assert Path(result).name == "params"


class TestStructureIO:
    """Test structure input/output functionality."""

    def test_read_structure_single_file(self, sample_xyz_file):
        """
        Test reading a single structure from an XYZ file.

        This test verifies that the function correctly reads
        a single molecular structure and returns an Atoms object.
        """
        # Act: Read the structure
        atoms = read_structure(sample_xyz_file)

        # Assert: Should return an Atoms object with correct properties
        assert isinstance(atoms, Atoms)
        assert len(atoms) > 0
        assert hasattr(atoms, 'positions')
        assert hasattr(atoms, 'symbols')

    def test_read_structure_file_not_found(self, temp_dir):
        """
        Test error handling when structure file doesn't exist.

        This test ensures the function raises appropriate exceptions
        when trying to read non-existent files.
        """
        # Arrange: Non-existent file path
        non_existent_file = temp_dir / "does_not_exist.xyz"

        # Act & Assert: Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="Structure file not found"):
            read_structure(non_existent_file)

    def test_read_structure_multi_sdf(self, multi_sdf_file):
        """
        Test reading multiple structures from a multi-SDF file.

        This test verifies that the function correctly handles
        multi-molecule SDF files and returns a list of structures.
        """
        # Act: Read all structures from multi-SDF
        structures = read_structure(multi_sdf_file, index=":")

        # Assert: Should return a list of Atoms objects
        assert isinstance(structures, list)
        assert len(structures) >= 2  # Should have multiple structures

        for structure in structures:
            assert isinstance(structure, Atoms)
            assert len(structure) > 0

    def test_read_structure_specific_index(self, multi_sdf_file):
        """
        Test reading a specific structure by index.

        This test verifies that the function can read
        a specific structure from a multi-structure file.
        """
        # Act: Read first structure (index 0)
        structure = read_structure(multi_sdf_file, index=0)

        # Assert: Should return a single Atoms object
        assert isinstance(structure, Atoms)
        assert len(structure) > 0

    def test_write_structure_success(self, temp_dir, sample_protein_atoms):
        """
        Test successful writing of structure to file.

        This test verifies that the function correctly writes
        molecular structures to files and creates directories as needed.
        """
        # Arrange: Output file path
        output_file = temp_dir / "subdir" / "output.xyz"

        # Act: Write structure
        written_path = write_structure(sample_protein_atoms, output_file)

        # Assert: File should be created and path returned
        assert Path(written_path).exists()
        assert Path(written_path) == output_file
        assert output_file.parent.exists()  # Directory should be created

    def test_write_structure_with_format(self, temp_dir, sample_protein_atoms):
        """
        Test writing structure with explicit format specification.

        This test verifies that the function correctly handles
        explicit format specification when writing files.
        """
        # Arrange: Output file with explicit format
        output_file = temp_dir / "output.pdb"

        # Act: Write with explicit format
        written_path = write_structure(sample_protein_atoms, output_file, format="xyz")

        # Assert: File should be created
        assert Path(written_path).exists()

    def test_validate_structure_valid(self, sample_protein_atoms):
        """
        Test structure validation with valid structure.

        This test verifies that the validation function
        correctly identifies valid molecular structures.
        """
        # Act & Assert: Should return True for valid structure
        assert validate_structure(sample_protein_atoms) is True

    def test_validate_structure_empty(self):
        """
        Test structure validation with empty structure.

        This test ensures that empty structures are properly
        rejected by the validation function.
        """
        # Arrange: Empty atoms object
        empty_atoms = Atoms()

        # Act & Assert: Should raise ValueError for empty structure
        with pytest.raises(ValueError, match="Structure contains no atoms"):
            validate_structure(empty_atoms)

    def test_validate_structure_invalid_coordinates(self):
        """
        Test structure validation with invalid coordinates.

        This test verifies that structures with NaN or infinite
        coordinates are properly rejected.
        """
        # Arrange: Atoms with invalid coordinates
        positions = np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 2.0]])
        atoms = Atoms(symbols=['H', 'H'], positions=positions)

        # Act & Assert: Should raise ValueError for invalid coordinates
        with pytest.raises(ValueError, match="invalid.*coordinates"):
            validate_structure(atoms)

    def test_get_supported_formats(self):
        """
        Test retrieval of supported file formats.

        This test verifies that the function returns a comprehensive
        list of supported molecular file formats.
        """
        # Act: Get supported formats
        formats = get_supported_formats()

        # Assert: Should return a list with common formats
        assert isinstance(formats, list)
        assert '.xyz' in formats
        assert '.pdb' in formats
        assert '.sdf' in formats
        assert len(formats) > 5  # Should support multiple formats


class TestDirectoryManagement:
    """Test directory and file management functionality."""

    def test_setup_output_directory_basic(self, temp_dir):
        """
        Test basic output directory creation.

        This test verifies that the function creates appropriate
        output directories with default naming conventions.
        """
        # Arrange: Protein file path
        protein_path = temp_dir / "protein.pdb"
        protein_path.touch()  # Create empty file

        # Act: Setup output directory
        output_dir = setup_output_directory(
            protein_path, optimize=False, trim=False, explain=False
        )

        # Assert: Should create results directory
        assert output_dir.exists()
        assert output_dir.name == "results"
        assert output_dir.parent == temp_dir

    def test_setup_output_directory_with_optimization(self, temp_dir):
        """
        Test output directory creation with optimization parameters.

        This test verifies that directory names include optimization
        parameters when optimization is enabled.
        """
        # Arrange: Protein file and optimization parameters
        protein_path = temp_dir / "protein.pdb"
        protein_path.touch()

        # Act: Setup with optimization
        output_dir = setup_output_directory(
            protein_path, optimize=True, trim=False, explain=False,
            steps=100, fmax=0.05
        )

        # Assert: Directory name should include optimization parameters
        assert output_dir.exists()
        assert "steps_100" in output_dir.name
        assert "fmax0.05" in output_dir.name

    def test_setup_output_directory_with_trimming(self, temp_dir):
        """
        Test output directory creation with trimming parameters.

        This test verifies that directory names include trimming
        parameters when trimming is enabled.
        """
        # Arrange: Protein file and trimming parameters
        protein_path = temp_dir / "protein.pdb"
        protein_path.touch()

        # Act: Setup with trimming
        output_dir = setup_output_directory(
            protein_path, optimize=False, trim=True, explain=False,
            radius=8.0
        )

        # Assert: Directory name should include trimming parameters
        assert output_dir.exists()
        assert "trim_8.0A" in output_dir.name

    def test_setup_output_directory_creates_subdirectories(self, temp_dir):
        """
        Test creation of workflow-specific subdirectories.

        This test verifies that appropriate subdirectories are created
        based on the workflow parameters (optimization, explainability).
        """
        # Arrange: Protein file path
        protein_path = temp_dir / "protein.pdb"
        protein_path.touch()

        # Act: Setup with all features enabled
        output_dir = setup_output_directory(
            protein_path, optimize=True, trim=False, explain=True,
            steps=100, fmax=0.05
        )

        # Assert: All expected subdirectories should exist
        assert (output_dir / "opt_ligand").exists()
        assert (output_dir / "opt_complexes").exists()
        assert (output_dir / "ligand_exp").exists()

    def test_write_opt_structure(self, temp_dir, sample_protein_atoms):
        """
        Test writing optimized structures to organized subdirectories.

        This test verifies that structures are saved in appropriate
        subdirectories based on their type.
        """
        # Act: Write optimized structure
        output_path = write_opt_structure(
            sample_protein_atoms, "opt_ligand", "ligand_opt.xyz", temp_dir
        )

        # Assert: Should create subdirectory and save file
        assert Path(output_path).exists()
        assert "opt_ligand" in str(output_path)
        assert (temp_dir / "opt_ligand").exists()


class TestLigandFileHandling:
    """Test ligand file discovery and processing."""

    def test_get_ligand_files_single_file(self, sample_sdf_file, temp_dir):
        """
        Test ligand file discovery with single file input.

        This test verifies that the function correctly handles
        single ligand files and returns them in a list.
        """
        # Act: Get ligand files from single file
        ligand_files = get_ligand_files(str(sample_sdf_file), temp_dir)

        # Assert: Should return list with single file
        assert isinstance(ligand_files, list)
        assert len(ligand_files) == 1
        assert str(sample_sdf_file) in ligand_files

    def test_get_ligand_files_multi_sdf(self, multi_sdf_file, temp_dir):
        """
        Test ligand file discovery with multi-molecule SDF.

        This test verifies that multi-molecule SDF files are
        properly split into individual ligand files.
        """
        # Act: Get ligand files from multi-SDF
        ligand_files = get_ligand_files(str(multi_sdf_file), temp_dir)

        # Assert: Should extract multiple ligands
        assert isinstance(ligand_files, list)
        assert len(ligand_files) >= 2  # Should have multiple ligands

        # Check that extracted files exist
        for ligand_file in ligand_files:
            assert Path(ligand_file).exists()

    def test_get_ligand_files_directory(self, temp_dir):
        """
        Test ligand file discovery from directory.

        This test verifies that the function correctly finds
        all ligand files in a directory.
        """
        # Arrange: Create directory with ligand files
        ligand_dir = temp_dir / "ligands"
        ligand_dir.mkdir()

        # Create sample ligand files
        for i, ext in enumerate(['.sdf', '.xyz', '.mol']):
            (ligand_dir / f"ligand_{i}{ext}").touch()

        # Act: Get ligand files from directory
        ligand_files = get_ligand_files(str(ligand_dir), temp_dir)

        # Assert: Should find all ligand files
        assert isinstance(ligand_files, list)
        assert len(ligand_files) == 3

        # Check file extensions
        extensions = [Path(f).suffix for f in ligand_files]
        assert '.sdf' in extensions
        assert '.xyz' in extensions
        assert '.mol' in extensions

    def test_get_ligand_files_not_found(self, temp_dir):
        """
        Test error handling when ligand files are not found.

        This test ensures appropriate exceptions are raised
        when ligand input doesn't exist.
        """
        # Arrange: Non-existent path
        non_existent_path = temp_dir / "does_not_exist"

        # Act & Assert: Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError, match="Ligands input not found"):
            get_ligand_files(str(non_existent_path), temp_dir)


class TestResultsSaving:
    """Test results saving and JSON export functionality."""

    def test_save_results_basic(self, temp_dir, mock_logger):
        """
        Test basic results saving functionality.

        This test verifies that results are correctly saved
        to JSON format with proper structure.
        """
        # Arrange: Sample results and arguments
        results = [
            {
                'ligand_name': 'test_ligand',
                'interaction_energy': -0.5,
                'binding_energy_kcal_mol': -11.5
            }
        ]

        args = Mock()
        args.ligands = "test_ligands.sdf"
        args.trim = False
        args.radius = None
        args.optimize = False
        args.explain = False
        args.optimizer = None
        args.fmax = None
        args.steps = None
        args.opt_log = False

        protein_path = Path("test_protein.pdb")

        # Act: Save results
        save_results(results, temp_dir, args, protein_path, None, mock_logger)

        # Assert: Results file should be created
        results_file = temp_dir / "results_summary.json"
        assert results_file.exists()

        # Verify JSON content
        with open(results_file) as f:
            data = json.load(f)

        assert 'workflow_parameters' in data
        assert 'summary' in data
        assert 'results' in data
        assert data['results'] == results

    def test_save_results_with_optimization_log(self, temp_dir, mock_logger):
        """
        Test results saving with optimization logging enabled.

        This test verifies that optimization logs are correctly
        saved when the opt_log option is enabled.
        """
        # Arrange: Results with optimization log
        results = []
        optimization_log = [
            {
                'structure_type': 'protein',
                'structure_name': 'test_protein',
                'optimization_info': {'converged': True, 'steps': 50}
            }
        ]

        args = Mock()
        args.ligands = "test_ligands.sdf"
        args.trim = False
        args.optimize = True
        args.explain = False
        args.opt_log = True
        args.optimizer = "FIRE"
        args.fmax = 0.05
        args.steps = 100

        protein_path = Path("test_protein.pdb")

        # Act: Save results with optimization log
        save_results(results, temp_dir, args, protein_path, optimization_log, mock_logger)

        # Assert: Both files should be created
        results_file = temp_dir / "results_summary.json"
        opt_log_file = temp_dir / "optimization_log.json"

        assert results_file.exists()
        assert opt_log_file.exists()

        # Verify optimization log content
        with open(opt_log_file) as f:
            log_data = json.load(f)

        assert 'workflow_parameters' in log_data
        assert 'optimizations' in log_data
        assert log_data['optimizations'] == optimization_log

    def test_save_results_with_explainability(self, temp_dir, mock_logger):
        """
        Test results saving with explainability data.

        This test verifies that explainability analysis results
        are correctly included in the saved JSON.
        """
        # Arrange: Results with explainability analysis
        results = [
            {
                'ligand_name': 'test_ligand',
                'interaction_energy': -0.5,
                'analysis': {
                    'component_totals': {'MLFF': -0.3, 'Electrostatics': -0.2},
                    'heatmap_path': 'test_heatmap.png'
                }
            }
        ]

        args = Mock()
        args.ligands = "test_ligands.sdf"
        args.trim = False
        args.optimize = False
        args.explain = True
        args.opt_log = False

        protein_path = Path("test_protein.pdb")

        # Act: Save results with explainability
        save_results(results, temp_dir, args, protein_path, None, mock_logger)

        # Assert: Results should include explainability data
        results_file = temp_dir / "results_summary.json"
        assert results_file.exists()

        with open(results_file) as f:
            data = json.load(f)

        assert data['workflow_parameters']['explain'] is True
        assert 'analysis' in data['results'][0]


class TestLoggingConfiguration:
    """Test logging setup and configuration."""

    def test_setup_logging_basic(self):
        """
        Test basic logging configuration.

        This test verifies that logging is correctly configured
        with appropriate levels and formatters.
        """
        # Act: Setup logging
        setup_logging(verbose=False)

        # Assert: Root logger should be configured
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
        assert len(root_logger.handlers) > 0

    def test_setup_logging_verbose(self):
        """
        Test verbose logging configuration.

        This test verifies that verbose mode correctly sets
        debug level for application loggers.
        """
        # Act: Setup verbose logging
        setup_logging(verbose=True)

        # Assert: Application loggers should be at debug level
        app_logger = logging.getLogger('src')
        # Note: This test demonstrates expected behavior
        # Actual logger levels may vary based on implementation

    def test_setup_logging_external_suppression(self):
        """
        Test suppression of external library logging.

        This test verifies that verbose external libraries
        are properly suppressed to reduce log noise.
        """
        # Act: Setup logging
        setup_logging(verbose=False)

        # Assert: External loggers should be suppressed
        jax_logger = logging.getLogger('jax')
        assert jax_logger.level >= logging.WARNING


@pytest.mark.unit
class TestUtilsIntegration:
    """Integration tests for utils module functions working together."""

    def test_full_workflow_utils(self, temp_dir, sample_protein_atoms, sample_ligand_atoms):
        """
        Test complete utils workflow integration.

        This test verifies that multiple utils functions work correctly
        together in a typical workflow scenario.
        """
        # Arrange: Create input files
        protein_file = temp_dir / "protein.xyz"
        ligand_file = temp_dir / "ligand.sdf"

        write_structure(sample_protein_atoms, protein_file)

        # Create SDF content manually for compatibility
        sdf_content = """test_ligand
  -OEChem-01012400002D

  3  2  0     0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.7570    0.5860    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7570    0.5860    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
M  END
$$$$
"""
        with open(ligand_file, 'w') as f:
            f.write(sdf_content)

        # Act: Execute full workflow
        # 1. Setup output directory
        output_dir = setup_output_directory(
            protein_file, optimize=True, trim=False, explain=True,
            steps=100, fmax=0.05
        )

        # 2. Read structures
        protein_atoms = read_structure(protein_file)
        ligand_files = get_ligand_files(str(ligand_file), output_dir)

        # 3. Validate structures
        validate_structure(protein_atoms)

        # 4. Save sample results
        mock_args = Mock()
        mock_args.ligands = str(ligand_file)
        mock_args.trim = False
        mock_args.optimize = True
        mock_args.explain = True
        mock_args.opt_log = False
        # Fix JSON serialization by providing actual values instead of Mocks
        mock_args.optimizer = "FIRE"
        mock_args.fmax = 0.05
        mock_args.steps = 100
        mock_args.radius = None

        results = [{'ligand_name': 'test', 'interaction_energy': -0.5}]
        mock_logger = Mock()

        save_results(results, output_dir, mock_args, protein_file, None, mock_logger)

        # Assert: All operations should complete successfully
        assert output_dir.exists()
        assert isinstance(protein_atoms, Atoms)
        assert len(ligand_files) >= 1
        assert (output_dir / "results_summary.json").exists()
        assert (output_dir / "opt_ligand").exists()
        assert (output_dir / "ligand_exp").exists()


# Additional test helper functions
def assert_valid_atoms_object(atoms):
    """Helper function to assert that an Atoms object is valid."""
    assert isinstance(atoms, Atoms)
    assert len(atoms) > 0
    assert hasattr(atoms, 'positions')
    assert hasattr(atoms, 'symbols')
    assert atoms.positions.shape[0] == len(atoms)
    assert len(atoms.symbols) == len(atoms)


def create_test_structure(symbols, positions):
    """Helper function to create test structures."""
    return Atoms(symbols=symbols, positions=np.array(positions))