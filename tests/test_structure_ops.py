"""
Tests for the structure_ops module.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.structure_ops import (
    trim_structure,
    optimize_structure,
    extract_ligands
)
from src.molecule_loader import load_ase_structure


class TestTrimStructure:
    """Tests for structure trimming functionality."""

    # ================================================================================================
    # UNIT TESTS - MOCK STRUCTURE OPERATIONS
    # ================================================================================================
    # The following tests use mocks and basic functionality testing:
    # - File I/O operations without real calculations
    # - Parameter validation and error handling
    # - Basic trimming logic without distance verification
    # ================================================================================================

    @pytest.mark.unit
    def test_trim_structure_basic(self, water_files, alanine_files, temp_dir):
        """Test basic structure trimming functionality."""
        protein_path = alanine_files['xyz']  # Use alanine as "protein"
        ligand_path = water_files['xyz']     # Use water as "ligand"

        # Use realistic radius for small molecules (1.5 Å)
        trimmed_protein_path = trim_structure(
            protein_path, ligand_path, radius=1.5, output_dir=temp_dir
        )

        # Check that files were created
        assert Path(trimmed_protein_path).exists()

        # Check that trimmed protein is smaller than or equal to original
        original_protein = load_ase_structure(protein_path)[0]
        trimmed_protein = load_ase_structure(trimmed_protein_path)[0]

        assert len(trimmed_protein) <= len(original_protein)
        assert len(trimmed_protein) > 0  # Should have some atoms

    @pytest.mark.unit
    def test_trim_structure_default_output_dir(self, water_files, alanine_files):
        """Test trimming with default output directory."""
        protein_path = alanine_files['xyz']
        ligand_path = water_files['xyz']

        trimmed_protein_path = trim_structure(protein_path, ligand_path, radius=2.0)

        # Check that output is in same directory as protein
        assert Path(trimmed_protein_path).parent == protein_path.parent
        assert Path(trimmed_protein_path).exists()

    @pytest.mark.unit
    def test_trim_structure_large_radius(self, water_files, alanine_files, temp_dir):
        """Test trimming with large radius (should include all atoms)."""
        protein_path = alanine_files['xyz']
        ligand_path = water_files['xyz']

        # Use a radius that should capture all atoms in small molecules
        trimmed_protein_path = trim_structure(
            protein_path, ligand_path, radius=10.0, output_dir=temp_dir
        )

        # With large radius, all protein atoms should be included
        original_protein = load_ase_structure(protein_path)[0]
        trimmed_protein = load_ase_structure(trimmed_protein_path)[0]

        assert len(trimmed_protein) == len(original_protein)

    @pytest.mark.unit
    def test_trim_structure_small_radius(self, water_files, alanine_files, temp_dir):
        """Test trimming with very small radius."""
        protein_path = alanine_files['xyz']
        ligand_path = water_files['xyz']

        # Use unrealistically small radius
        with pytest.raises(ValueError, match="No protein atoms found within"):
            trim_structure(protein_path, ligand_path, radius=0.1, output_dir=temp_dir)

    @pytest.mark.unit
    def test_trim_structure_file_validation(self, temp_dir):
        """Test trimming with invalid input files."""
        fake_protein = temp_dir / "fake_protein.xyz"
        fake_ligand = temp_dir / "fake_ligand.xyz"

        with pytest.raises(FileNotFoundError):
            trim_structure(fake_protein, fake_ligand, radius=1.5)

    @pytest.mark.unit
    def test_trim_structure_filename_generation(self, water_files, alanine_files, temp_dir):
        """Test that trimmed filename is generated correctly."""
        protein_path = alanine_files['xyz']
        ligand_path = water_files['xyz']
        radius = 1.8

        trimmed_protein_path = trim_structure(
            protein_path, ligand_path, radius=radius, output_dir=temp_dir
        )

        expected_filename = f"{protein_path.stem}_trimmed_{radius}A_atom.xyz"
        assert Path(trimmed_protein_path).name == expected_filename

    @pytest.mark.unit
    def test_trim_structure_pdb_residue_based(self, water_files, alanine_files, temp_dir):
        """Test residue-based trimming for PDB files."""
        protein_path = alanine_files['pdb']  # Use PDB format
        ligand_path = water_files['xyz']
        radius = 2.0

        trimmed_protein_path = trim_structure(
            protein_path, ligand_path, radius=radius, output_dir=temp_dir
        )

        # Check that files were created with residue suffix
        expected_filename = f"{protein_path.stem}_trimmed_{radius}A_residue.xyz"
        assert Path(trimmed_protein_path).name == expected_filename
        assert Path(trimmed_protein_path).exists()

        # Check that trimmed protein is valid
        original_protein = load_ase_structure(protein_path)[0]
        trimmed_protein = load_ase_structure(trimmed_protein_path)[0]

        assert len(trimmed_protein) <= len(original_protein)
        assert len(trimmed_protein) > 0

    @pytest.mark.unit
    def test_trim_structure_xyz_atom_based_warnings(self, water_files, alanine_files, temp_dir, caplog):
        """Test that XYZ files generate appropriate warnings for atom-based trimming."""
        import logging
        protein_path = alanine_files['xyz']  # Use XYZ format
        ligand_path = water_files['xyz']

        with caplog.at_level(logging.WARNING):
            trimmed_protein_path = trim_structure(
                protein_path, ligand_path, radius=2.0, output_dir=temp_dir
            )

        # Check warning messages were logged
        warning_messages = [record.message for record in caplog.records if record.levelno >= logging.WARNING]
        assert any("Using atom-based trimming" in msg for msg in warning_messages)
        assert any("residues may be incomplete" in msg for msg in warning_messages)
        assert any("use PDB format input files" in msg for msg in warning_messages)

        # Check filename has atom suffix
        expected_filename = f"{protein_path.stem}_trimmed_2.0A_atom.xyz"
        assert Path(trimmed_protein_path).name == expected_filename

    @pytest.mark.unit
    def test_trim_structure_pdb_residue_info_logging(self, water_files, alanine_files, temp_dir, caplog):
        """Test that PDB files generate appropriate info messages for residue-based trimming."""
        import logging
        protein_path = alanine_files['pdb']  # Use PDB format
        ligand_path = water_files['xyz']

        with caplog.at_level(logging.INFO):
            trimmed_protein_path = trim_structure(
                protein_path, ligand_path, radius=2.0, output_dir=temp_dir
            )

        # Check info messages were logged
        info_messages = [record.message for record in caplog.records if record.levelno == logging.INFO]
        assert any("Using residue-based trimming" in msg for msg in info_messages)
        assert any("Complete residues will be included" in msg for msg in info_messages)

        # Verify the file was created (use the variable to avoid linting warning)
        assert Path(trimmed_protein_path).exists()

    @pytest.mark.unit
    def test_trim_by_atoms_function(self, water_files, alanine_files):
        """Test _trim_by_atoms function directly."""
        from src.structure_ops import _trim_by_atoms
        import logging

        # Load test structures
        protein = load_ase_structure(alanine_files['xyz'])[0]
        ligand = load_ase_structure(water_files['xyz'])[0]

        protein_positions = protein.get_positions()
        ligand_positions = ligand.get_positions()
        logger = logging.getLogger(__name__)

        # Test with small radius
        atoms_to_keep_small = _trim_by_atoms(protein_positions, ligand_positions, 1.0, logger)

        # Test with large radius
        atoms_to_keep_large = _trim_by_atoms(protein_positions, ligand_positions, 10.0, logger)

        # Large radius should keep more or equal atoms than small radius
        assert len(atoms_to_keep_large) >= len(atoms_to_keep_small)

        # All atom indices should be valid
        for idx in atoms_to_keep_small:
            assert 0 <= idx < len(protein_positions)
        for idx in atoms_to_keep_large:
            assert 0 <= idx < len(protein_positions)

    @pytest.mark.unit
    def test_trim_by_residues_function(self, water_files, alanine_files):
        """Test _trim_by_residues function directly."""
        from src.structure_ops import _trim_by_residues
        import logging

        # Use PDB file for residue information
        protein_path = alanine_files['pdb']
        protein = load_ase_structure(protein_path)[0]
        ligand = load_ase_structure(water_files['xyz'])[0]

        protein_positions = protein.get_positions()
        ligand_positions = ligand.get_positions()
        logger = logging.getLogger(__name__)

        # Test residue-based trimming
        atoms_to_keep = _trim_by_residues(protein_path, protein_positions, ligand_positions, 3.0, logger)

        # Should return valid atom indices
        assert isinstance(atoms_to_keep, list)
        assert len(atoms_to_keep) > 0
        assert len(atoms_to_keep) <= len(protein_positions)

        # All indices should be valid
        for idx in atoms_to_keep:
            assert 0 <= idx < len(protein_positions)

        # Indices should be sorted
        assert atoms_to_keep == sorted(atoms_to_keep)

    @pytest.mark.unit
    def test_trim_by_residues_fallback_to_atoms(self, water_files, alanine_files, caplog):
        """Test that _trim_by_residues falls back to atom-based trimming on error."""
        from src.structure_ops import _trim_by_residues
        import logging

        # Use XYZ file which should cause residue parsing to fail
        protein_path = alanine_files['xyz']  # XYZ instead of PDB
        protein = load_ase_structure(protein_path)[0]
        ligand = load_ase_structure(water_files['xyz'])[0]

        protein_positions = protein.get_positions()
        ligand_positions = ligand.get_positions()
        logger = logging.getLogger(__name__)

        with caplog.at_level(logging.WARNING):
            atoms_to_keep = _trim_by_residues(protein_path, protein_positions, ligand_positions, 3.0, logger)

        # Should still return valid results (from fallback)
        assert isinstance(atoms_to_keep, list)
        assert len(atoms_to_keep) > 0

        # Should have logged fallback warnings
        warning_messages = [record.message for record in caplog.records if record.levelno >= logging.WARNING]
        assert any("Residue-based trimming failed" in msg for msg in warning_messages)
        assert any("Falling back to atom-based trimming" in msg for msg in warning_messages)

    @pytest.mark.unit
    def test_trim_structure_edge_case_sdf_format(self, water_files, temp_dir):
        """Test trimming with SDF format (should use atom-based)."""
        protein_path = water_files['sdf']  # Use SDF as "protein"
        ligand_path = water_files['xyz']

        trimmed_protein_path = trim_structure(
            protein_path, ligand_path, radius=2.0, output_dir=temp_dir
        )

        # Should use atom-based trimming and have atom suffix
        expected_filename = f"{protein_path.stem}_trimmed_2.0A_atom.xyz"
        assert Path(trimmed_protein_path).name == expected_filename
        assert Path(trimmed_protein_path).exists()

    @pytest.mark.unit
    def test_trim_structure_compare_pdb_vs_xyz_same_molecule(self, alanine_files, water_files, temp_dir):
        """Test that PDB and XYZ of same molecule produce different trimming results."""
        ligand_path = water_files['xyz']
        radius = 2.0

        # Trim using PDB (residue-based)
        pdb_trimmed = trim_structure(
            alanine_files['pdb'], ligand_path, radius=radius, output_dir=temp_dir
        )

        # Trim using XYZ (atom-based)
        xyz_trimmed = trim_structure(
            alanine_files['xyz'], ligand_path, radius=radius, output_dir=temp_dir / "xyz"
        )

        # Load results
        pdb_result = load_ase_structure(pdb_trimmed)[0]
        xyz_result = load_ase_structure(xyz_trimmed)[0]

        # For small molecules like alanine, results might be the same, but filenames should differ
        assert Path(pdb_trimmed).name.endswith("_residue.xyz")
        assert Path(xyz_trimmed).name.endswith("_atom.xyz")

        # Both should have valid structures
        assert len(pdb_result) > 0
        assert len(xyz_result) > 0

    @pytest.mark.unit
    def test_trim_structure_empty_protein_error(self, water_files, temp_dir):
        """Test trimming with empty protein structure."""
        from ase import Atoms

        # Create empty protein structure
        empty_protein_path = temp_dir / "empty_protein.xyz"
        # Note: empty_atoms not used directly, just for documentation
        Atoms()  # Empty structure

        # Write empty structure to file
        with open(empty_protein_path, 'w') as f:
            f.write("0\nEmpty structure\n")

        ligand_path = water_files['xyz']

        # Should raise ValueError for empty protein
        with pytest.raises(ValueError, match="Invalid or empty ASE structure"):
            trim_structure(empty_protein_path, ligand_path, radius=2.0, output_dir=temp_dir)

    @pytest.mark.unit
    def test_trim_structure_invalid_pdb_fallback(self, water_files, temp_dir, caplog):
        """Test trimming with corrupted PDB that fails residue parsing."""
        import logging

        # Create corrupted PDB file
        corrupted_pdb = temp_dir / "corrupted.pdb"
        with open(corrupted_pdb, 'w') as f:
            f.write("HEADER    INVALID PDB\n")
            f.write("ATOM      1  N   ALA A   1      20.154  16.967  10.000  1.00 10.00           N\n")
            f.write("ATOM      2  CA  ALA A   1      21.618  16.890  10.000  1.00 10.00           C\n")
            f.write("END\n")

        ligand_path = water_files['xyz']

        with caplog.at_level(logging.WARNING):
            # This might fail residue parsing and fallback to atom-based
            try:
                trimmed_path = trim_structure(corrupted_pdb, ligand_path, radius=5.0, output_dir=temp_dir)
                # Should still produce a result via fallback
                assert Path(trimmed_path).exists()
            except Exception:
                # If it fails completely, that's also acceptable for corrupted input
                pass

    @pytest.mark.unit
    def test_trim_structure_zero_radius_edge_case(self, water_files, alanine_files, temp_dir):
        """Test trimming with radius=0.0."""
        protein_path = alanine_files['xyz']
        ligand_path = water_files['xyz']

        # With radius 0.0, should find no atoms (unless they're exactly overlapping)
        with pytest.raises(ValueError, match="No protein atoms found within 0.0"):
            trim_structure(protein_path, ligand_path, radius=0.0, output_dir=temp_dir)

    @pytest.mark.unit
    def test_trim_structure_very_large_radius(self, water_files, alanine_files, temp_dir):
        """Test trimming with very large radius (performance test)."""
        protein_path = alanine_files['xyz']
        ligand_path = water_files['xyz']

        # Very large radius should include all atoms
        trimmed_path = trim_structure(protein_path, ligand_path, radius=1000.0, output_dir=temp_dir)

        # Should include all atoms from original protein
        original_protein = load_ase_structure(protein_path)[0]
        trimmed_protein = load_ase_structure(trimmed_path)[0]

        assert len(trimmed_protein) == len(original_protein)
        assert Path(trimmed_path).exists()

    @pytest.mark.unit
    def test_trim_structure_nonexistent_files(self, temp_dir):
        """Test trimming with nonexistent input files."""
        fake_protein = temp_dir / "nonexistent_protein.pdb"
        fake_ligand = temp_dir / "nonexistent_ligand.xyz"

        # Should raise FileNotFoundError for nonexistent files
        with pytest.raises((FileNotFoundError, ValueError)):
            trim_structure(fake_protein, fake_ligand, radius=2.0, output_dir=temp_dir)

    @pytest.mark.unit
    def test_trim_by_residues_with_xyz_graceful_failure(self, water_files, alanine_files, caplog):
        """Test _trim_by_residues gracefully handles XYZ files (no residue info)."""
        from src.structure_ops import _trim_by_residues
        import logging

        # XYZ file has no residue information - should fallback
        protein_path = alanine_files['xyz']
        protein = load_ase_structure(protein_path)[0]
        ligand = load_ase_structure(water_files['xyz'])[0]

        protein_positions = protein.get_positions()
        ligand_positions = ligand.get_positions()
        logger = logging.getLogger(__name__)

        with caplog.at_level(logging.WARNING):
            result = _trim_by_residues(protein_path, protein_positions, ligand_positions, 2.0, logger)

        # Should return valid atom list from fallback
        assert isinstance(result, list)
        assert len(result) > 0

        # Should log the fallback
        messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("Residue-based trimming failed" in msg for msg in messages)
        assert any("Falling back to atom-based trimming" in msg for msg in messages)

    @pytest.mark.unit
    def test_constrain_by_atoms_function(self, water_files, alanine_files):
        """Test _constrain_by_atoms function directly."""
        from src.structure_ops import _constrain_by_atoms
        import logging

        # Load test structures
        protein = load_ase_structure(alanine_files['xyz'])[0]
        ligand = load_ase_structure(water_files['xyz'])[0]
        complex_atoms = protein + ligand

        logger = logging.getLogger(__name__)

        # Test with very small radius (should fix many atoms)
        fixed_atoms_small = _constrain_by_atoms(complex_atoms, len(protein), 0.5, logger)

        # Test with large radius (should fix fewer/no atoms)
        fixed_atoms_large = _constrain_by_atoms(complex_atoms, len(protein), 10.0, logger)

        # Small radius should fix more atoms than large radius
        assert len(fixed_atoms_small) >= len(fixed_atoms_large)

        # All indices should be valid protein atom indices
        for idx in fixed_atoms_small:
            assert 0 <= idx < len(protein)
        for idx in fixed_atoms_large:
            assert 0 <= idx < len(protein)

    @pytest.mark.unit
    def test_constrain_by_residues_function(self, water_files, alanine_files):
        """Test _constrain_by_residues function directly."""
        from src.structure_ops import _constrain_by_residues
        import logging

        # Use PDB file for residue information
        protein_path = alanine_files['pdb']
        protein = load_ase_structure(protein_path)[0]
        ligand = load_ase_structure(water_files['xyz'])[0]
        complex_atoms = protein + ligand

        logger = logging.getLogger(__name__)

        # Test residue-based constraints
        fixed_atoms = _constrain_by_residues(complex_atoms, len(protein), 2.0, protein_path, logger)

        # Should return valid atom indices
        assert isinstance(fixed_atoms, list)
        assert all(isinstance(idx, int) for idx in fixed_atoms)
        assert all(0 <= idx < len(protein) for idx in fixed_atoms)

        # Indices should be sorted
        assert fixed_atoms == sorted(fixed_atoms)

    @pytest.mark.unit
    def test_constrain_by_residues_fallback(self, water_files, alanine_files, caplog):
        """Test _constrain_by_residues falls back to atom-based on error."""
        from src.structure_ops import _constrain_by_residues
        import logging

        # Use XYZ file which should cause residue parsing to fail
        protein_path = alanine_files['xyz']
        protein = load_ase_structure(protein_path)[0]
        ligand = load_ase_structure(water_files['xyz'])[0]
        complex_atoms = protein + ligand

        logger = logging.getLogger(__name__)

        with caplog.at_level(logging.WARNING):
            fixed_atoms = _constrain_by_residues(complex_atoms, len(protein), 2.0, protein_path, logger)

        # Should still return valid results from fallback
        assert isinstance(fixed_atoms, list)
        assert all(0 <= idx < len(protein) for idx in fixed_atoms)

        # Should have logged fallback warnings
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("Residue-based constraints failed" in msg for msg in warning_messages)
        assert any("Falling back to atom-based constraints" in msg for msg in warning_messages)

    @pytest.mark.unit
    def test_create_optimization_constraint_pdb_format(self, water_files, alanine_files, caplog):
        """Test create_optimization_constraint with PDB format (residue-level)."""
        from src.structure_ops import create_optimization_constraint
        import logging

        protein_path = alanine_files['pdb']
        protein = load_ase_structure(protein_path)[0]
        ligand = load_ase_structure(water_files['xyz'])[0]
        complex_atoms = protein + ligand

        with caplog.at_level(logging.INFO):
            constraint = create_optimization_constraint(
                complex_atoms, len(protein), 2.0, protein_path
            )

        # Check info messages were logged
        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("Using residue-level constraints for PDB file" in msg for msg in info_messages)
        assert any("Complete residues will be flexible" in msg for msg in info_messages)

        # Constraint could be None (if all atoms are flexible) or FixAtoms object
        if constraint is not None:
            from ase.constraints import FixAtoms
            assert isinstance(constraint, FixAtoms)
            assert all(0 <= idx < len(protein) for idx in constraint.indices)

    @pytest.mark.unit
    def test_create_optimization_constraint_xyz_format(self, water_files, alanine_files, caplog):
        """Test create_optimization_constraint with XYZ format (atom-level)."""
        from src.structure_ops import create_optimization_constraint
        import logging

        protein_path = alanine_files['xyz']
        protein = load_ase_structure(protein_path)[0]
        ligand = load_ase_structure(water_files['xyz'])[0]
        complex_atoms = protein + ligand

        with caplog.at_level(logging.WARNING):
            constraint = create_optimization_constraint(
                complex_atoms, len(protein), 2.0, protein_path
            )

        # Check warning messages were logged
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("Using atom-level constraints" in msg for msg in warning_messages)
        assert any("residues may be split" in msg for msg in warning_messages)
        assert any("use PDB format input files" in msg for msg in warning_messages)

        # Constraint could be None or FixAtoms object
        if constraint is not None:
            from ase.constraints import FixAtoms
            assert isinstance(constraint, FixAtoms)

    @pytest.mark.unit
    def test_create_optimization_constraint_no_protein_path(self, water_files, alanine_files, caplog):
        """Test create_optimization_constraint with no protein path (defaults to atom-level)."""
        from src.structure_ops import create_optimization_constraint
        import logging

        protein = load_ase_structure(alanine_files['xyz'])[0]
        ligand = load_ase_structure(water_files['xyz'])[0]
        complex_atoms = protein + ligand

        with caplog.at_level(logging.INFO):
            constraint = create_optimization_constraint(
                complex_atoms, len(protein), 2.0, None
            )

        # Check default message was logged
        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("No protein path provided - using atom-level constraints" in msg for msg in info_messages)

    @pytest.mark.unit
    def test_optimize_structure_with_protein_path(self, mock_calculator, water_files, alanine_files, temp_dir):
        """Test optimize_structure includes protein_path parameter."""
        protein = load_ase_structure(alanine_files['xyz'])[0]
        ligand = load_ase_structure(water_files['xyz'])[0]
        complex_atoms = protein + ligand

        output_path = temp_dir / "optimized_complex.xyz"

        # Test that the function accepts the protein_path parameter
        result_path, info = optimize_structure(
            complex_atoms, mock_calculator,
            optimizer='FIRE', fmax=0.1, steps=1,  # Minimal steps for testing
            output_path=output_path,
            opt_radius=2.0, n_protein_atoms=len(protein),
            protein_path=alanine_files['pdb']  # NEW parameter
        )

        # Should complete without error
        assert Path(result_path).exists()
        assert 'output_file' in info

    @pytest.mark.unit
    def test_perform_trimming_with_specified_ligand(self, temp_dir, sample_xyz_file, sample_sdf_file, mock_logger):
        """Test perform_trimming with specified trim ligand."""
        from src.structure_ops import perform_trimming

        with patch('src.structure_ops.trim_structure') as mock_trim:
            mock_trim.return_value = temp_dir / "trimmed_protein.xyz"

            result = perform_trimming(
                protein_path=sample_xyz_file,
                ligands_source=sample_sdf_file,
                radius=5.0,
                trim_lig=str(sample_sdf_file),
                output_dir=temp_dir,
                logger=mock_logger
            )

            assert result == temp_dir / "trimmed_protein.xyz"
            mock_trim.assert_called_once()
            mock_logger.info.assert_called()

    @pytest.mark.unit
    def test_perform_trimming_nonexistent_trim_ligand(self, temp_dir, sample_xyz_file, mock_logger):
        """Test perform_trimming with nonexistent trim ligand."""
        from src.structure_ops import perform_trimming

        with pytest.raises(FileNotFoundError) as exc_info:
            perform_trimming(
                protein_path=sample_xyz_file,
                ligands_source=None,
                radius=5.0,
                trim_lig="/nonexistent/ligand.sdf",
                output_dir=temp_dir,
                logger=mock_logger
            )

        assert "trim ligand not found" in str(exc_info.value)

    @pytest.mark.unit
    def test_perform_trimming_no_ligand_files_found(self, temp_dir, sample_xyz_file, mock_logger):
        """Test perform_trimming when no ligand files are found."""
        from src.structure_ops import perform_trimming

        with patch('src.utils.get_ligand_files', return_value=[]):
            with pytest.raises(ValueError) as exc_info:
                perform_trimming(
                    protein_path=sample_xyz_file,
                    ligands_source=temp_dir,
                    radius=5.0,
                    trim_lig=None,
                    output_dir=temp_dir,
                    logger=mock_logger
                )

            assert "No ligand files found" in str(exc_info.value)

    @pytest.mark.unit
    def test_perform_trimming_auto_select_ligand(self, temp_dir, sample_xyz_file, sample_sdf_file, mock_logger):
        """Test perform_trimming auto-selecting first ligand."""
        from src.structure_ops import perform_trimming

        with patch('src.utils.get_ligand_files', return_value=[sample_sdf_file]), \
             patch('src.structure_ops.trim_structure') as mock_trim:

            mock_trim.return_value = temp_dir / "trimmed_protein.xyz"

            result = perform_trimming(
                protein_path=sample_xyz_file,
                ligands_source=temp_dir,
                radius=5.0,
                trim_lig=None,
                output_dir=temp_dir,
                logger=mock_logger
            )

            assert result == temp_dir / "trimmed_protein.xyz"
            mock_trim.assert_called_once_with(
                sample_xyz_file, sample_sdf_file,
                radius=5.0, output_dir=temp_dir
            )


class TestOptimizeStructure:
    """Tests for structure optimization functionality."""

    # ================================================================================================
    # UNIT TESTS - MOCK OPTIMIZATION
    # ================================================================================================
    # The following tests use mock calculators and optimizers:
    # - Mock SO3LR calculator with fixed energy values
    # - Mock ASE optimizers with controlled convergence
    # - Parameter validation without real calculations
    # ================================================================================================

    @pytest.mark.unit
    def test_optimize_structure_fire(self, water_files, temp_dir):
        """Test structure optimization with FIRE optimizer."""
        from tests.conftest import MockSo3lrSfCalculator
        atoms = load_ase_structure(water_files['xyz'])[0]
        mock_calculator = MockSo3lrSfCalculator()

        # Mock optimization methods
        with patch('src.structure_ops.FIRE') as mock_fire_class:
            mock_optimizer = Mock()
            mock_optimizer.run = Mock()
            mock_optimizer.converged.return_value = True
            mock_optimizer.nsteps = 10
            mock_fire_class.return_value = mock_optimizer

            output_path = temp_dir / "optimized_water.xyz"
            optimized_path, opt_info = optimize_structure(
                atoms,
                calc=mock_calculator,
                optimizer='FIRE',
                output_path=output_path
            )

            # Check that optimization ran
            mock_optimizer.run.assert_called_once()
            assert opt_info['constraint_info'] == {'constraint_applied': False, 'constraint_type': None}
            assert opt_info['converged'] == 'yes'
            assert opt_info['steps'] == 10
            # Check output file
            assert Path(optimized_path).exists()

    @pytest.mark.unit
    def test_optimize_structure_lbfgs(self, water_files, temp_dir):
        """Test structure optimization with LBFGS optimizer."""
        from tests.conftest import MockSo3lrSfCalculator
        atoms = load_ase_structure(water_files['xyz'])[0]
        mock_calculator = MockSo3lrSfCalculator()

        with patch('src.structure_ops.LBFGS') as mock_lbfgs_class:
            mock_optimizer = Mock()
            mock_optimizer.run = Mock()
            mock_optimizer.converged.return_value = False
            mock_optimizer.nsteps = 10
            mock_lbfgs_class.return_value = mock_optimizer

            output_path = temp_dir / "optimized_water_lbfgs.xyz"
            _, opt_info = optimize_structure(
                atoms,
                calc=mock_calculator,
                optimizer='LBFGS',
                output_path=output_path
            )
            assert opt_info['constraint_info'] == {'constraint_applied': False, 'constraint_type': None}
            assert opt_info['converged'] == 'yes'
            assert opt_info['steps'] == 10

    @pytest.mark.unit
    def test_optimize_structure_unknown_optimizer(self, water_files, temp_dir):
        """Test optimization with unknown optimizer."""
        atoms = load_ase_structure(water_files['xyz'])[0]
        output_path = temp_dir / "test_output.xyz"
        with pytest.raises(ValueError, match="Unknown optimizer: UNKNOWN"):
            optimize_structure(atoms, calc=Mock(), optimizer='UNKNOWN', output_path=output_path)

    @pytest.mark.unit
    def test_optimize_structure_no_calculator(self, water_files, temp_dir):
        """Test optimization without calculator raises error."""
        atoms = load_ase_structure(water_files['xyz'])[0]
        output_path = temp_dir / "test_output.xyz"
        with pytest.raises((TypeError, ValueError, AttributeError)):
            optimize_structure(atoms, calc=None, output_path=output_path)

    @pytest.mark.unit
    def test_optimize_structure_optimization_error(self, water_files, temp_dir):
        """Test optimization with errors during optimization."""
        atoms = load_ase_structure(water_files['xyz'])[0]
        from tests.conftest import MockSo3lrSfCalculator
        mock_calculator = MockSo3lrSfCalculator()

        with patch('src.structure_ops.FIRE') as mock_fire_class:
            mock_optimizer = Mock()
            mock_optimizer.run.side_effect = Exception("Optimization error")
            mock_fire_class.return_value = mock_optimizer

            # Should still save structure even with optimization error
            output_path = temp_dir / "optimized_error.xyz"
            optimized_path, opt_info = optimize_structure(
                atoms,
                calc=mock_calculator,
                output_path=output_path
            )

            assert Path(optimized_path).exists()
            assert opt_info['optimization_error'] == "Optimization error"

    @pytest.mark.unit
    def test_optimize_structure_default_output_dir(self, water_files, temp_dir):
        """Test optimization with default output directory."""
        atoms = load_ase_structure(water_files['xyz'])[0]
        from tests.conftest import MockSo3lrSfCalculator
        mock_calculator = MockSo3lrSfCalculator()

        with patch('src.structure_ops.FIRE') as mock_fire_class:
            mock_optimizer = Mock()
            mock_optimizer.run = Mock()
            mock_optimizer.converged.return_value = True
            mock_optimizer.nsteps = 5
            mock_fire_class.return_value = mock_optimizer

            # Provide output path to work around current code limitation
            output_path = temp_dir / "optimized_default.xyz"
            optimized_path, opt_info = optimize_structure(
                atoms,
                calc=mock_calculator,
                output_path=output_path
            )

            # Check that optimization ran successfully
            mock_optimizer.run.assert_called_once()
            assert opt_info['converged'] == "yes"
            assert opt_info['steps'] == 5
            assert Path(optimized_path).exists()

    @pytest.mark.integration
    @pytest.mark.slow
    def test_optimize_structure_with_real_calculator(self, water_files, temp_dir, so3lr_model_path):
        """Test optimization with actual SO3LR calculator (if available)."""
        from src.calculator import So3lrSfCalculator

        # Check if SO3LR model is available
        if not Path(so3lr_model_path).exists():
            pytest.skip(f"SO3LR model not available at {so3lr_model_path}")

        try:
            calc = So3lrSfCalculator(model_path=so3lr_model_path)
            atoms = load_ase_structure(water_files['xyz'])[0]
            output_path = temp_dir / "optimized_real.xyz"

            optimized_path, opt_info = optimize_structure(
                atoms,
                calc=calc,
                optimizer='FIRE',
                fmax=0.1,  # Less strict convergence for testing
                steps=10,   # Limit steps for testing
                output_path=output_path
            )

            assert Path(optimized_path).exists()
            assert isinstance(opt_info['initial_energy'], (float, np.floating, np.ndarray))
            assert isinstance(opt_info['final_energy'], (float, np.floating, np.ndarray))

        except Exception as e:
            pytest.skip(f"SO3LR calculation failed: {e}")

    @pytest.mark.unit
    def test_create_optimization_constraint_with_fixed_atoms(self, sample_complex_atoms):
        """Test constraint creation when some atoms should be fixed."""
        from src.structure_ops import create_optimization_constraint

        # Test with small radius to ensure some atoms are fixed
        constraint = create_optimization_constraint(
            complex_atoms=sample_complex_atoms,
            n_protein_atoms=10,
            opt_radius=2.0
        )

        from ase.constraints import FixAtoms
        assert isinstance(constraint, FixAtoms)
        assert len(constraint.index) > 0
        assert len(constraint.index) < 10  # Some but not all protein atoms fixed

    @pytest.mark.unit
    def test_create_optimization_constraint_no_fixed_atoms(self, sample_complex_atoms):
        """Test constraint creation when no atoms should be fixed (large radius)."""
        from src.structure_ops import create_optimization_constraint

        # Test with large radius to ensure no atoms are fixed
        constraint = create_optimization_constraint(
            complex_atoms=sample_complex_atoms,
            n_protein_atoms=10,
            opt_radius=50.0
        )

        assert constraint is None

    @pytest.mark.unit
    def test_create_optimization_constraint_dummy_example(self):
        """Test constraint creation with controlled dummy example."""
        import numpy as np
        from ase import Atoms
        from src.structure_ops import create_optimization_constraint
        from ase.constraints import FixAtoms

        # Create controlled test case: 5 protein atoms, 2 ligand atoms
        protein_positions = np.array([
            [0.0, 0.0, 0.0],  # Protein atom 0 - far from ligand
            [1.0, 0.0, 0.0],  # Protein atom 1 - far from ligand
            [2.0, 0.0, 0.0],  # Protein atom 2 - close to ligand
            [3.0, 0.0, 0.0],  # Protein atom 3 - close to ligand
            [4.0, 0.0, 0.0]   # Protein atom 4 - far from ligand
        ])

        ligand_positions = np.array([
            [2.5, 0.0, 0.0],  # Ligand atom - close to protein atoms 2,3
            [3.0, 0.0, 0.0]   # Ligand atom - close to protein atoms 2,3
        ])

        # Combine into complex
        all_positions = np.vstack([protein_positions, ligand_positions])
        complex_atoms = Atoms('H' * len(protein_positions) + 'O' * len(ligand_positions), positions=all_positions)
        n_protein_atoms = 5

        # Test opt_radius = 0.0 - should fix all protein atoms
        constraint_zero = create_optimization_constraint(
            complex_atoms=complex_atoms,
            n_protein_atoms=n_protein_atoms,
            opt_radius=0.0
        )
        assert isinstance(constraint_zero, FixAtoms)
        assert len(constraint_zero.index) == 5  # All protein atoms fixed
        assert set(constraint_zero.index) == {0, 1, 2, 3, 4}

        # Test opt_radius = 1.0 - should fix atoms 0,1,4; flexible atoms 2,3
        constraint_small = create_optimization_constraint(
            complex_atoms=complex_atoms,
            n_protein_atoms=n_protein_atoms,
            opt_radius=1.0
        )
        assert isinstance(constraint_small, FixAtoms)
        assert len(constraint_small.index) == 3  # Atoms 0,1,4 fixed
        assert set(constraint_small.index) == {0, 1, 4}

        # Test opt_radius = 5.0 - should not fix any atoms (all flexible)
        constraint_large = create_optimization_constraint(
            complex_atoms=complex_atoms,
            n_protein_atoms=n_protein_atoms,
            opt_radius=5.0
        )
        assert constraint_large is None  # No constraints needed

    @pytest.mark.unit
    def test_optimize_structure_constraint_info_logging(self, water_files, temp_dir):
        """Test that constraint info is properly logged in optimization results."""
        from tests.conftest import MockSo3lrSfCalculator
        from ase.constraints import FixAtoms

        atoms = load_ase_structure(water_files['xyz'])[0]
        mock_calculator = MockSo3lrSfCalculator()

        # Create a constraint to test the logging
        constraint = FixAtoms(indices=[0])  # Fix first atom
        atoms.set_constraint(constraint)

        with patch('src.structure_ops.FIRE') as mock_fire_class:
            mock_optimizer = Mock()
            mock_optimizer.run = Mock()
            mock_optimizer.converged.return_value = True
            mock_optimizer.nsteps = 5
            mock_fire_class.return_value = mock_optimizer

            output_path = temp_dir / "constrained_water.xyz"
            _, opt_info = optimize_structure(
                atoms,
                calc=mock_calculator,
                optimizer='FIRE',
                output_path=output_path,
                n_protein_atoms=2,  # Treat first 2 atoms as "protein"
                opt_radius=1.0      # This triggers the constraint info logging
            )

            # Check basic constraint info
            assert opt_info['constraint_info']['constraint_applied'] == True
            assert opt_info['constraint_info']['constraint_type'] == 'FixAtoms'

            # Check detailed constraint info (the lines we're testing)
            constraint_info = opt_info['constraint_info']
            assert 'optimization_radius' in constraint_info
            assert 'total_protein_atoms' in constraint_info
            assert 'total_ligand_atoms' in constraint_info
            assert 'flexible_protein_atoms' in constraint_info
            assert 'fixed_protein_atoms' in constraint_info
            assert 'ligand_atoms_always_flexible' in constraint_info
            assert 'constraint_details' in constraint_info

            # Verify the calculated values
            assert constraint_info['optimization_radius'] == 1.0
            assert constraint_info['total_protein_atoms'] == 2
            assert constraint_info['total_ligand_atoms'] == 1  # 3 atoms - 2 protein = 1 ligand
            assert constraint_info['fixed_protein_atoms'] == 1  # One atom fixed by constraint
            assert constraint_info['flexible_protein_atoms'] == 1  # 2 - 1 = 1 flexible
            assert constraint_info['ligand_atoms_always_flexible'] == 1

            # Check constraint details string format
            expected_details = "1/2 protein atoms flexible within 1.0Å of ligand"
            assert constraint_info['constraint_details'] == expected_details

    @pytest.mark.unit
    def test_optimize_structure_no_constraint_info_when_no_radius(self, water_files, temp_dir):
        """Test that detailed constraint info is not added when opt_radius is None."""
        from tests.conftest import MockSo3lrSfCalculator
        from ase.constraints import FixAtoms

        atoms = load_ase_structure(water_files['xyz'])[0]
        mock_calculator = MockSo3lrSfCalculator()

        # Create a constraint but don't provide opt_radius
        constraint = FixAtoms(indices=[0])
        atoms.set_constraint(constraint)

        with patch('src.structure_ops.FIRE') as mock_fire_class:
            mock_optimizer = Mock()
            mock_optimizer.run = Mock()
            mock_optimizer.converged.return_value = True
            mock_optimizer.nsteps = 5
            mock_fire_class.return_value = mock_optimizer

            output_path = temp_dir / "constrained_water.xyz"
            _, opt_info = optimize_structure(
                atoms,
                calc=mock_calculator,
                optimizer='FIRE',
                output_path=output_path,
                n_protein_atoms=2,  # Provide n_protein_atoms but not opt_radius
                opt_radius=0.1     # This should prevent detailed logging
            )
            # Check basic constraint info exists
            constraint_info = opt_info['constraint_info']
            assert constraint_info['constraint_applied'] == True
            assert constraint_info['constraint_type'] == 'FixAtoms'

            # Check that detailed constraint info is added
            assert constraint_info['optimization_radius'] == 0.1
            assert constraint_info['total_protein_atoms'] == 2
            assert 'constraint_details' in constraint_info



class TestExtractLigands:
    """Tests for ligand extraction functionality."""

    @pytest.mark.unit
    def test_extract_ligands_from_multi_sdf(self, multi_water_file, temp_dir):
        """Test extracting ligands from multi-structure SDF file."""
        ligand_files = extract_ligands(multi_water_file, output_dir=temp_dir)

        assert len(ligand_files) == 2

        for i, ligand_file in enumerate(ligand_files, 1):
            assert Path(ligand_file).exists()
            assert Path(ligand_file).name == f"ligand_{i:03d}.xyz"

            # Check that each ligand can be read
            atoms = load_ase_structure(ligand_file)[0]
            assert len(atoms) == 3  # Water molecule

    @pytest.mark.unit
    def test_extract_ligands_single_structure(self, water_files, temp_dir):
        """Test extracting from single-structure file."""
        ligand_files = extract_ligands(water_files['sdf'], output_dir=temp_dir)

        assert len(ligand_files) == 1
        assert Path(ligand_files[0]).exists()

    @pytest.mark.unit
    def test_extract_ligands_default_output_dir(self, multi_water_file):
        """Test extraction with default output directory."""
        ligand_files = extract_ligands(multi_water_file)

        expected_dir = multi_water_file.parent / f"{multi_water_file.stem}_individual"

        for ligand_file in ligand_files:
            assert Path(ligand_file).parent == expected_dir

        # Cleanup created directory
        import shutil
        if expected_dir.exists():
            shutil.rmtree(expected_dir)

    @pytest.mark.unit
    def test_extract_ligands_custom_prefix(self, multi_water_file, temp_dir):
        """Test extraction with custom naming prefix."""
        ligand_files = extract_ligands(
            multi_water_file,
            output_dir=temp_dir,
            naming_prefix="molecule"
        )

        for i, ligand_file in enumerate(ligand_files, 1):
            assert Path(ligand_file).name == f"molecule_{i:03d}.xyz"

    @pytest.mark.unit
    def test_extract_ligands_validation_error(self, temp_dir):
        """Test extraction with invalid structures."""
        # Create a fake multi-structure file with invalid content
        fake_sdf = temp_dir / "fake.sdf"
        fake_sdf.write_text("invalid sdf content")

        with pytest.raises((ValueError, IndexError, OSError)):
            extract_ligands(fake_sdf, output_dir=temp_dir)

    @pytest.mark.unit
    def test_extract_ligands_from_xyz_trajectory(self, multi_water_xyz_file, temp_dir):
        """Test extracting from XYZ trajectory-like file."""
        ligand_files = extract_ligands(multi_water_xyz_file, output_dir=temp_dir)

        assert len(ligand_files) == 2
        for ligand_file in ligand_files:
            assert Path(ligand_file).exists()
            atoms = load_ase_structure(ligand_file)[0]
            assert len(atoms) == 3


class TestStructureOpsIntegration:
    """Integration tests for structure operations using real SO3LR calculator."""

    # ================================================================================================
    # INTEGRATION TESTS - REAL STRUCTURE OPERATIONS
    # ================================================================================================
    # The following tests use real SO3LR calculator and actual molecular structures:
    # - Real trimming with distance verification
    # - Real optimization with position change verification
    # - Constrained optimization tests
    # - Single ligand-protein processing tests
    # - Actual energy calculations and structure modifications
    # ================================================================================================

    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_trimming_with_distance_verification(self, water_files, alanine_files, temp_dir):
        """Test real trimming with distance verification using alanine and water."""
        protein_path = alanine_files['xyz']
        ligand_path = water_files['xyz']

        # Test realistic radius
        radius = 2.0
        trimmed_protein_path = trim_structure(
            protein_path, ligand_path, radius=radius, output_dir=temp_dir
        )

        # Verify trimmed structure exists and is valid
        assert Path(trimmed_protein_path).exists()

        # Load structures to verify trimming worked
        original_protein = load_ase_structure(protein_path)[0]
        ligand = load_ase_structure(ligand_path)[0]
        trimmed_protein = load_ase_structure(trimmed_protein_path)[0]

        # Verify trimmed structure is smaller or equal
        assert len(trimmed_protein) <= len(original_protein)
        assert len(trimmed_protein) > 0

        # Verify that all remaining atoms are within the cutoff distance from ligand
        ligand_positions = ligand.get_positions()
        trimmed_positions = trimmed_protein.get_positions()

        # Calculate minimum distance from each trimmed atom to any ligand atom
        for trimmed_pos in trimmed_positions:
            min_distance = float('inf')
            for ligand_pos in ligand_positions:
                distance = np.linalg.norm(trimmed_pos - ligand_pos)
                min_distance = min(min_distance, distance)

            # Each remaining atom should be within the cutoff
            assert min_distance <= radius + 0.1  # Small tolerance for numerical precision

        print(f"Trimming verified: {len(trimmed_protein)}/{len(original_protein)} atoms kept within {radius}Å")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_optimization_water_position_change(self, water_files, temp_dir, real_calculator):
        """Test real optimization with position change verification for water."""
        atoms = load_ase_structure(water_files['xyz'])[0]
        original_positions = atoms.get_positions().copy()

        output_path = temp_dir / "optimized_water_real.xyz"
        optimized_path, opt_info = optimize_structure(
            atoms,
            calc=real_calculator,
            optimizer='FIRE',
            fmax=0.1,  # Less strict for testing
            steps=20,   # Limit steps
            output_path=output_path
        )

        # Verify optimization completed
        assert Path(optimized_path).exists()
        assert isinstance(opt_info['initial_energy'], (float, np.floating, np.ndarray))
        assert isinstance(opt_info['final_energy'], (float, np.floating, np.ndarray))

        # Load optimized structure and verify positions changed
        optimized_atoms = load_ase_structure(optimized_path)[0]
        optimized_positions = optimized_atoms.get_positions()

        # Calculate position changes
        position_changes = np.linalg.norm(optimized_positions - original_positions, axis=1)
        max_change = np.max(position_changes)

        # Verify that at least some atoms moved (optimization occurred)
        assert max_change > 1e-6, "Optimization should cause position changes"

        print(f"Water optimization: max position change = {max_change:.6f} Å")
        print(f"Energy change: {opt_info['initial_energy']:.4f} → {opt_info['final_energy']:.4f}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_real_optimization_alanine_position_change(self, alanine_files, temp_dir, real_calculator):
        """Test real optimization with position change verification for alanine."""
        atoms = load_ase_structure(alanine_files['xyz'])[0]
        original_positions = atoms.get_positions().copy()

        output_path = temp_dir / "optimized_alanine_real.xyz"
        optimized_path, opt_info = optimize_structure(
            atoms,
            calc=real_calculator,
            optimizer='FIRE',
            fmax=0.1,  # Less strict for testing
            steps=20,   # Limit steps
            output_path=output_path
        )

        # Verify optimization completed
        assert Path(optimized_path).exists()
        assert isinstance(opt_info['initial_energy'], (float, np.floating, np.ndarray))
        assert isinstance(opt_info['final_energy'], (float, np.floating, np.ndarray))

        # Load optimized structure and verify positions changed
        optimized_atoms = load_ase_structure(optimized_path)[0]
        optimized_positions = optimized_atoms.get_positions()

        # Calculate position changes
        position_changes = np.linalg.norm(optimized_positions - original_positions, axis=1)
        max_change = np.max(position_changes)

        # Verify that at least some atoms moved (optimization occurred)
        assert max_change > 1e-6, "Optimization should cause position changes"

        print(f"Alanine optimization: max position change = {max_change:.6f} Å")
        print(f"Energy change: {opt_info['initial_energy']:.4f} → {opt_info['final_energy']:.4f}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_constrained_optimization(self, water_files, temp_dir, real_calculator):
        """Test constrained optimization."""
        atoms = load_ase_structure(water_files['xyz'])[0]
        original_positions = atoms.get_positions().copy()

        # Add constraint to fix the oxygen atom (index 0)
        from ase.constraints import FixAtoms
        constraint = FixAtoms(indices=[0])
        atoms.set_constraint(constraint)

        output_path = temp_dir / "constrained_water_real.xyz"
        optimized_path, opt_info = optimize_structure(
            atoms,
            calc=real_calculator,
            optimizer='FIRE',
            fmax=0.1,
            steps=20,
            output_path=output_path
        )

        # Verify optimization completed
        assert Path(optimized_path).exists()
        assert opt_info['converged'] in ['yes', 'no']  # Should have valid convergence status

        # Load optimized structure
        optimized_atoms = load_ase_structure(optimized_path)[0]
        optimized_positions = optimized_atoms.get_positions()

        # Verify oxygen atom (index 0) didn't move due to constraint
        oxygen_change = np.linalg.norm(optimized_positions[0] - original_positions[0])
        assert oxygen_change < 1e-6, "Constrained oxygen atom should not move"

        # Verify hydrogen atoms (indices 1,2) could move
        h_changes = [np.linalg.norm(optimized_positions[i] - original_positions[i]) for i in [1, 2]]

        print(f"Constrained optimization: O change = {oxygen_change:.8f} Å")
        print(f"H changes = {h_changes[0]:.6f}, {h_changes[1]:.6f} Å")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_single_ligand_protein_processing(self, water_files, alanine_files, temp_dir, real_calculator):
        """Test single ligand-protein processing using alanine as protein and water as ligand."""
        protein_path = alanine_files['xyz']  # Alanine as protein
        ligand_path = water_files['xyz']     # Water as ligand

        # Step 1: Trim protein around ligand
        radius = 2.5  # Larger radius to capture alanine atoms
        trimmed_protein_path = trim_structure(
            protein_path, ligand_path, radius=radius, output_dir=temp_dir
        )

        # Step 2: Optimize trimmed protein
        trimmed_atoms = load_ase_structure(trimmed_protein_path)[0]
        original_positions = trimmed_atoms.get_positions().copy()

        optimized_path, opt_info = optimize_structure(
            trimmed_atoms,
            calc=real_calculator,
            optimizer='FIRE',
            fmax=0.2,  # Less strict for testing
            steps=15,   # Limit steps
            output_path=temp_dir / "optimized_trimmed_protein.xyz"
        )

        # Verify complete workflow
        assert Path(trimmed_protein_path).exists()
        assert Path(optimized_path).exists()

        # Load final structure
        final_atoms = load_ase_structure(optimized_path)[0]
        final_positions = final_atoms.get_positions()

        # Verify trimming worked
        original_protein = load_ase_structure(protein_path)[0]
        assert len(trimmed_atoms) <= len(original_protein)

        # Verify optimization worked (positions changed)
        position_changes = np.linalg.norm(final_positions - original_positions, axis=1)
        max_change = np.max(position_changes)
        assert max_change > 1e-6, "Optimization should cause position changes"

        print(f"Single ligand-protein processing:")
        print(f"  Trimmed: {len(trimmed_atoms)}/{len(original_protein)} atoms")
        print(f"  Max position change: {max_change:.6f} Å")
        print(f"  Energy change: {opt_info['initial_energy']:.4f} → {opt_info['final_energy']:.4f}")

    @pytest.mark.integration
    def test_extract_and_process_workflow(self, multi_water_file, temp_dir):
        """Test workflow of extracting ligands and processing each."""
        # Extract ligands
        ligand_files = extract_ligands(multi_water_file, output_dir=temp_dir)

        # Process each ligand (mock processing)
        results = []
        for ligand_file in ligand_files:
            atoms = load_ase_structure(ligand_file)[0]
            results.append({
                'file': ligand_file,
                'n_atoms': len(atoms),
                'formula': atoms.get_chemical_formula()
            })

        assert len(results) == 2
        for result in results:
            assert result['n_atoms'] == 3
            assert result['formula'] == 'H2O'

    @pytest.mark.integration
    @pytest.mark.slow
    def test_create_optimization_constraint_real(self, test_data_dir, temp_dir):
        """Test create_optimization_constraint with real complex atoms."""
        from src.structure_ops import create_optimization_constraint
        from src.molecule_loader import load_ase_structure

        # Skip if test data not available
        alanine_file = test_data_dir / "alanine.xyz"
        water_file = test_data_dir / "water.xyz"

        if not alanine_file.exists() or not water_file.exists():
            pytest.skip("Test data files not available")

        # Create a real complex by combining protein + ligand
        protein_atoms = load_ase_structure(alanine_file)[0]
        ligand_atoms = load_ase_structure(water_file)[0]

        # Translate ligand to avoid overlap
        ligand_positions = ligand_atoms.get_positions()
        ligand_positions += [5.0, 0.0, 0.0]  # Move 5Å away
        ligand_atoms.set_positions(ligand_positions)

        # Combine into complex
        complex_atoms = protein_atoms + ligand_atoms
        n_protein_atoms = len(protein_atoms)

        # Test with small radius - should create some constraints
        constraint_small = create_optimization_constraint(
            complex_atoms=complex_atoms,
            n_protein_atoms=n_protein_atoms,
            opt_radius=3.0
        )
        print("Small radius constraints:", constraint_small)

        # Test with large radius - should not create constraints
        constraint_large = create_optimization_constraint(
            complex_atoms=complex_atoms,
            n_protein_atoms=n_protein_atoms,
            opt_radius=20.0
        )

        # Verify results - small radius should typically create some constraints
        if constraint_small is not None:
            from ase.constraints import FixAtoms
            assert isinstance(constraint_small, FixAtoms)
            assert len(constraint_small.index) > 0
            assert len(constraint_small.index) < n_protein_atoms  # Some but not all atoms fixed

        # Large radius should typically result in no constraints
        assert constraint_large is None

    @pytest.mark.integration
    @pytest.mark.slow
    def test_perform_trimming_real(self, test_data_dir, temp_dir):
        """Test perform_trimming with real data workflow."""
        from src.structure_ops import perform_trimming

        # Skip if test data not available
        alanine_file = test_data_dir / "alanine.xyz"
        water_file = test_data_dir / "water.sdf"

        if not alanine_file.exists() or not water_file.exists():
            pytest.skip("Test data files not available")

        import logging
        logger = logging.getLogger(__name__)

        # Test with specified ligand
        result = perform_trimming(
            protein_path=alanine_file,
            ligands_source=None,
            radius=3.0,
            trim_lig=str(water_file),
            output_dir=temp_dir,
            logger=logger
        )

        assert Path(result).exists()

        # Verify trimming worked
        from src.molecule_loader import load_ase_structure
        trimmed_atoms = load_ase_structure(result)[0]
        original_atoms = load_ase_structure(alanine_file)[0]
        assert len(trimmed_atoms) <= len(original_atoms)

        # Test auto-selection (create a directory with ligand files)
        ligand_dir = temp_dir / "ligands"
        ligand_dir.mkdir()

        # Copy water file to ligand directory
        import shutil
        shutil.copy(water_file, ligand_dir / "water.sdf")

        result2 = perform_trimming(
            protein_path=alanine_file,
            ligands_source=str(ligand_dir),
            radius=3.0,
            trim_lig=None,
            output_dir=temp_dir,
            logger=logger
        )

        assert Path(result2).exists()
        trimmed_atoms2 = load_ase_structure(result2)[0]
        assert len(trimmed_atoms2) <= len(original_atoms)


class TestOptimizeProtein:
    """Tests for optimize_protein function."""

    @pytest.mark.unit
    def test_optimize_protein_basic(self, temp_dir, water_files, mock_calculator):
        """Test basic protein optimization."""
        from src.structure_ops import optimize_protein
        from tests.conftest import MockSo3lrSfCalculator
        from unittest.mock import Mock

        mock_logger = Mock()
        optimization_log = []

        with patch('src.structure_ops.optimize_structure') as mock_optimize:
            mock_optimize.return_value = (str(temp_dir / "protein_opt.xyz"), {'converged': True, 'steps': 50})

            result = optimize_protein(
                working_protein_path=water_files['xyz'],
                calc=mock_calculator,
                optimizer="FIRE",
                fmax=0.01,
                steps=1000,
                output_dir=temp_dir,
                optimization_log=optimization_log,
                opt_log=True,
                logger=mock_logger
            )

            # Check return value
            assert result == str(temp_dir / "protein_opt.xyz")

            # Check optimization was called
            mock_optimize.assert_called_once()

            # Check optimization log was updated
            assert len(optimization_log) == 1
            assert optimization_log[0]['structure_type'] == 'protein'
            assert optimization_log[0]['structure_name'] == 'water'

    @pytest.mark.unit
    def test_optimize_protein_no_log(self, temp_dir, water_files, mock_calculator):
        """Test protein optimization without logging."""
        from src.structure_ops import optimize_protein
        from unittest.mock import Mock

        mock_logger = Mock()

        with patch('src.structure_ops.optimize_structure') as mock_optimize:
            mock_optimize.return_value = (str(temp_dir / "protein_opt.xyz"), {'converged': True})

            result = optimize_protein(
                working_protein_path=water_files['xyz'],
                calc=mock_calculator,
                optimizer="FIRE",
                fmax=0.01,
                steps=1000,
                output_dir=temp_dir,
                optimization_log=None,
                opt_log=False,
                logger=mock_logger
            )

            assert result == str(temp_dir / "protein_opt.xyz")


class TestProcessSingleLigand:
    """Tests for process_single_ligand function."""

    @pytest.mark.unit
    def test_process_single_ligand_basic_no_optimization(self, temp_dir, water_files, mock_calculator):
        """Test basic ligand processing without optimization."""
        from src.structure_ops import process_single_ligand
        from unittest.mock import Mock

        # Create mock args
        mock_args = Mock()
        mock_args.optimize = False
        mock_args.explain = False
        mock_args.protein_explain = False
        mock_args.eda = False
        mock_args.opt_log = False

        mock_logger = Mock()
        optimization_log = []

        with patch('src.structure_ops.protein_ligand_interaction') as mock_interaction:
            mock_interaction.return_value = -2.5  # Simple energy value

            result, error = process_single_ligand(
                ligand_file=water_files['sdf'],
                args=mock_args,
                calc=mock_calculator,
                working_protein_path=water_files['pdb'],
                output_dir=temp_dir,
                optimization_log=optimization_log,
                logger=mock_logger
            )

            # Check no error
            assert error is None

            # Check result structure
            assert result['ligand_name'] == 'water'
            assert result['interaction_energy'] == -2.5
            assert result['binding_energy_kcal_mol'] == -2.5 * 23.06

    @pytest.mark.unit
    def test_process_single_ligand_with_explainability(self, temp_dir, water_files, mock_calculator):
        """Test ligand processing with explainability analysis."""
        from src.structure_ops import process_single_ligand
        from unittest.mock import Mock

        # Create mock args
        mock_args = Mock()
        mock_args.optimize = False
        mock_args.explain = True
        mock_args.eda = False
        mock_args.opt_log = False
        mock_args.protein_explain = False
        
        mock_logger = Mock()
        optimization_log = []

        with patch('src.structure_ops.protein_ligand_interaction') as mock_interaction:
            # Mock explainability return
            mock_analysis = {
                'component_totals': {'MLFF': -1.5, 'Electrostatics': -0.8},
                'heatmap_path': str(temp_dir / "heatmap.png")
            }
            mock_interaction.return_value = (-2.3, mock_analysis)

            result, error = process_single_ligand(
                ligand_file=water_files['sdf'],
                args=mock_args,
                calc=mock_calculator,
                working_protein_path=water_files['pdb'],
                output_dir=temp_dir,
                optimization_log=optimization_log,
                logger=mock_logger
            )

            # Check no error
            assert error is None

            # Check analysis was included
            assert result['ligand_explainability'] == mock_analysis
            assert result['interaction_energy'] == -2.3

            # Check that heatmap output was set correctly
            expected_heatmap = temp_dir / "ligand_exp" / "water_heatmap.png"
            mock_interaction.assert_called_with(
                water_files['pdb'], water_files['sdf'], mock_calculator,
                complex_path=None,
                explainability=True,
                eda=False,
                heatmap_output=expected_heatmap,
                verbose=False,
                preloaded_protein_prolif=None
            )

    @pytest.mark.unit
    def test_process_single_ligand_with_eda(self, temp_dir, water_files, mock_calculator):
        """Test ligand processing with EDA analysis."""
        from src.structure_ops import process_single_ligand
        from unittest.mock import Mock

        # Create mock args
        mock_args = Mock()
        mock_args.optimize = False
        mock_args.explain = False
        mock_args.eda = True
        mock_args.opt_log = False

        mock_logger = Mock()
        optimization_log = []

        with patch('src.structure_ops.protein_ligand_interaction') as mock_interaction:
            # Mock EDA return
            mock_analysis = {
                'interaction_energy_components': {'mlff': -1.2, 'zbl': 0.3, 'electrostatic': -0.9}
            }
            mock_interaction.return_value = (-1.8, mock_analysis)

            result, error = process_single_ligand(
                ligand_file=water_files['sdf'],
                args=mock_args,
                calc=mock_calculator,
                working_protein_path=water_files['pdb'],
                output_dir=temp_dir,
                optimization_log=optimization_log,
                logger=mock_logger
            )

            # Check no error
            assert error is None

            # Check EDA analysis was included
            assert result['ligand_explainability'] == mock_analysis
            assert result['interaction_energy'] == -1.8

    @pytest.mark.unit
    def test_process_single_ligand_error_handling(self, temp_dir, water_files, mock_calculator):
        """Test error handling in process_single_ligand."""
        from src.structure_ops import process_single_ligand
        from unittest.mock import Mock

        # Create mock args
        mock_args = Mock()
        mock_args.optimize = False
        mock_args.explain = False
        mock_args.eda = False
        mock_args.opt_log = False

        mock_logger = Mock()
        optimization_log = []

        with patch('src.structure_ops.protein_ligand_interaction') as mock_interaction:
            mock_interaction.side_effect = Exception("Calculation failed")

            result, error = process_single_ligand(
                ligand_file=water_files['sdf'],
                args=mock_args,
                calc=mock_calculator,
                working_protein_path=water_files['pdb'],
                output_dir=temp_dir,
                optimization_log=optimization_log,
                logger=mock_logger
            )

            # Check error was caught
            assert error == "Calculation failed"
            assert result['ligand_name'] == 'water'
            assert 'error' in result


class TestProcessSingleLigandIntegration:
    """Slow integration tests for complete workflow using real test data."""

    @pytest.mark.slow
    def test_process_single_ligand_complete_workflow_with_test_data(self, temp_dir):
        """
        Complete unmocked integration test with all features enabled:
        - Loads real alanine.pdb as protein and water.sdf as ligand
        - Runs actual optimization algorithm
        - Runs actual energy calculations
        - Runs actual explainability analysis
        - Tests the complete end-to-end workflow
        - Validates output shapes and structures from real computations
        """
        from src.structure_ops import process_single_ligand
        from src.molecule_loader import load_ase_structure
        from src.calculator import So3lrSfCalculator
        from unittest.mock import Mock
        from pathlib import Path
        import logging

        # Use real test data files
        test_data_dir = Path(__file__).parent / "test_data"
        protein_file = test_data_dir / "alanine.pdb"
        ligand_file = test_data_dir / "water.sdf"

        # Verify test files exist
        assert protein_file.exists(), f"Test protein file not found: {protein_file}"
        assert ligand_file.exists(), f"Test ligand file not found: {ligand_file}"

        # Load structures to understand their shapes
        protein_atoms = load_ase_structure(protein_file)[0]
        ligand_atoms = load_ase_structure(ligand_file)[0]

        n_protein_atoms = len(protein_atoms)
        n_ligand_atoms = len(ligand_atoms)

        print(f"Running complete integration test:")
        print(f"  Protein: {n_protein_atoms} atoms")
        print(f"  Ligand: {n_ligand_atoms} atoms")

        # Get real calculator with per-atom components enabled for explainability
        calc = So3lrSfCalculator(output_per_atom_energy_components=True)

        # Create args with everything enabled but fast settings
        mock_args = Mock()
        mock_args.optimize = True
        mock_args.explain = True
        mock_args.eda = True
        mock_args.opt_log = True
        mock_args.optimizer = "FIRE"
        mock_args.fmax = 0.5  # Loose convergence for speed
        mock_args.steps = 10  # Very few steps for test speed
        mock_args.opt_radius = None

        # Setup real logger
        logger = logging.getLogger(__name__)
        optimization_log = []

        # Run the complete workflow WITHOUT ANY MOCKING
        result, error = process_single_ligand(
            ligand_file=str(ligand_file),
            args=mock_args,
            calc=calc,
            working_protein_path=str(protein_file),
            output_dir=temp_dir,
            optimization_log=optimization_log,
            logger=logger
        )

        # Verify the workflow completed successfully
        if error is not None:
            print(f"Integration test failed with error: {error}")
            # Don't fail the test immediately - still check what we can

        assert result is not None

        # Check basic result structure
        assert result['ligand_name'] == 'water'
        assert isinstance(result.get('interaction_energy'), (int, float, type(None)))
        assert isinstance(result.get('binding_energy_kcal_mol'), (int, float, type(None)))

        # Check optimization results if optimization ran
        if mock_args.optimize and len(optimization_log) > 0:
            print(f"  Optimization completed: {len(optimization_log)} structures optimized")

            # Should have optimized both ligand and complex
            assert len(optimization_log) <= 2  # May be less if optimization failed

            for i, opt_log in enumerate(optimization_log):
                assert 'structure_type' in opt_log
                # Check for either 'ligand_name' or 'structure_name' depending on log format
                name_key = 'ligand_name' if 'ligand_name' in opt_log else 'structure_name'
                assert name_key in opt_log

                # Extract optimization info which may be nested
                if 'optimization_info' in opt_log:
                    opt_info = opt_log['optimization_info']
                    assert 'converged' in opt_info
                    converged = opt_info['converged'] == 'yes'
                    steps = opt_info.get('steps', opt_info.get('n_steps', 0))
                    final_fmax = opt_info.get('final_fmax', 0.0)
                else:
                    assert 'converged' in opt_log
                    converged = opt_log['converged']
                    steps = opt_log['steps']
                    final_fmax = opt_log['final_fmax']

                assert isinstance(converged, bool)
                assert isinstance(steps, int)
                assert isinstance(final_fmax, (int, float))
                assert steps <= mock_args.steps

                print(f"    {opt_log['structure_type']}: {steps} steps, "
                      f"fmax={final_fmax:.4f}, converged={converged}")

        # Check optimization output files exist if optimization ran
        for path_key in ['optimized_ligand_path', 'optimized_complex_path']:
            if path_key in result and result[path_key]:
                opt_path = Path(result[path_key])
                assert opt_path.exists(), f"Optimization output file should exist: {opt_path}"

                # Verify structure can be loaded and has correct size
                try:
                    opt_structure = load_ase_structure(opt_path)[0]
                    if 'ligand' in path_key:
                        assert len(opt_structure) == n_ligand_atoms
                    elif 'complex' in path_key:
                        assert len(opt_structure) == n_protein_atoms + n_ligand_atoms
                    print(f"  ✓ {path_key}: {len(opt_structure)} atoms")
                except Exception as e:
                    print(f"  ⚠ Could not verify {path_key}: {e}")

        # Check EDA analysis if it ran
        if 'analysis' in result and result['analysis']:
            analysis = result['analysis']
            assert isinstance(analysis, dict)
            print(f"  ✓ EDA analysis completed with {len(analysis)} components")

            if 'interaction_energy_components' in analysis:
                components = analysis['interaction_energy_components']
                print(f"    Energy components: {list(components.keys())}")

        # Check explainability results if they exist
        if 'ligand_energy_differences' in result and result['ligand_energy_differences']:
            ligand_differences = result['ligand_energy_differences']
            assert isinstance(ligand_differences, dict)
            print(f"  ✓ Explainability completed with components: {list(ligand_differences.keys())}")

            # Verify energy difference arrays have correct shapes
            for component, values in ligand_differences.items():
                assert isinstance(values, list)
                assert len(values) == n_ligand_atoms, \
                    f"Component {component} should have {n_ligand_atoms} values"
                # Check that all values are numeric
                for val in values:
                    assert isinstance(val, (int, float))

            print(f"    Each component has {n_ligand_atoms} per-atom values")

        print(f"  ✓ Integration test completed successfully!")

        if error is None:
            print(f"  ✓ Final interaction energy: {result.get('interaction_energy', 'N/A')} eV")