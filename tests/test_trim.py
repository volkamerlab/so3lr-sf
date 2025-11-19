"""
Tests for the trim module.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.trim import (
    trim_structure,
    perform_trimming,
    _trim_by_atoms,
    _trim_by_residues
)
from src.molecule_loader import load_ase_structure


class TestTrimStructure:
    """Tests for structure trimming functionality."""

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
    def test_trim_structure_pdb_residue_based(self, water_files, alanine_files, temp_dir):
        """Test residue-based trimming for PDB files."""
        protein_path = alanine_files['pdb']  # Use PDB format
        ligand_path = water_files['xyz']
        radius = 2.0

        trimmed_protein_path = trim_structure(
            protein_path, ligand_path, radius=radius, output_dir=temp_dir
        )

        # Check that files were created with residue suffix
        expected_filename = f"{protein_path.stem}_trimmed_{radius}A_residue.pdb"
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
    def test_trim_structure_small_radius(self, water_files, alanine_files, temp_dir):
        """Test trimming with very small radius."""
        protein_path = alanine_files['xyz']
        ligand_path = water_files['xyz']

        # Use unrealistically small radius
        with pytest.raises(ValueError, match="No protein atoms found within"):
            trim_structure(protein_path, ligand_path, radius=0.1, output_dir=temp_dir)

    @pytest.mark.unit
    def test_trim_by_atoms_function(self, water_files, alanine_files):
        """Test _trim_by_atoms function directly."""
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
    def test_perform_trimming_with_specified_ligand(self, temp_dir, sample_xyz_file, sample_sdf_file, mock_logger):
        """Test perform_trimming with specified trim ligand."""
        with patch('src.trim.trim_structure') as mock_trim:
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
    def test_perform_trimming_no_ligand_files_found(self, temp_dir, sample_xyz_file, mock_logger):
        """Test perform_trimming when no ligand files are found."""
        with patch('src.trim.get_ligand_files', return_value=[]):
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
        with patch('src.trim.get_ligand_files', return_value=[sample_sdf_file]), \
             patch('src.trim.trim_structure') as mock_trim:

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

    @pytest.mark.unit
    def test_trim_structure_unsupported_format(self, water_files, temp_dir):
        """Test trimming with SDF format (should raise error for unsupported format)."""
        protein_path = water_files['sdf']  # Use SDF as "protein"
        ligand_path = water_files['xyz']

        # SDF format is not supported for protein input
        with pytest.raises(ValueError, match="Unsupported protein file format for trimming: .sdf"):
            trim_structure(protein_path, ligand_path, radius=2.0, output_dir=temp_dir)

    @pytest.mark.unit
    def test_trim_structure_large_radius_includes_all(self, water_files, alanine_files, temp_dir):
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