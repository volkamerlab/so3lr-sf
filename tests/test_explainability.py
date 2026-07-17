"""
Tests for the explainability module.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile

from src.explainability import (
    generate_ligand_heatmap,
    generate_protein_interaction_heatmap,
    generate_energy_heatmap,
    _fp_interaction_mapping
)


class TestExplainabilityModule:
    """Tests for explainability functions."""

    @pytest.fixture
    def sample_ligand_energy_differences(self):
        """Sample ligand energy differences for testing (3 atoms for water molecule)."""
        return {
            'MLFF': np.array([0.1, -0.2, 0.3]),
            'Electrostatics': np.array([0.05, 0.1, -0.15]),
            'Dispersion': np.array([-0.01, -0.02, 0.03]),
            'Total': np.array([0.14, -0.12, 0.18])
        }

    @pytest.fixture
    def sample_protein_energy_differences(self):
        """Sample protein energy differences for testing."""
        return {
            'MLFF': np.array([0.2, -0.1, 0.15, -0.05, 0.1, 0.08, -0.12, 0.06, -0.03, 0.04]),
            'Electrostatics': np.array([0.1, 0.05, -0.08, 0.12, -0.06, 0.03, 0.09, -0.04, 0.07, -0.02]),
            'Dispersion': np.array([-0.02, 0.01, -0.03, 0.02, -0.01, 0.015, -0.025, 0.005, 0.01, -0.005]),
            'Total': np.array([0.28, -0.04, 0.04, 0.09, 0.03, 0.125, -0.055, 0.025, 0.047, 0.035])
        }

    @pytest.fixture
    def sample_atom_mappings(self):
        """Sample atom mappings for protein-ligand interactions."""
        return [
            {
                'frame': 0,
                'ligand_residue': 'UNL1',
                'protein_residue': 'ARG45.A',
                'interaction_type': 'HBDonor',
                'ligand_atoms': [0, 1],
                'protein_atoms': [120, 121, 122],
                'distance': 2.1
            },
            {
                'frame': 0,
                'ligand_residue': 'UNL1',
                'protein_residue': 'PHE89.A',
                'interaction_type': 'Hydrophobic',
                'ligand_atoms': [2, 3],
                'protein_atoms': [280, 281, 282, 283],
                'distance': 3.8
            },
            {
                'frame': 0,
                'ligand_residue': 'UNL1',
                'protein_residue': 'ASP67.A',
                'interaction_type': 'HBAcceptor',
                'ligand_atoms': [4],
                'protein_atoms': [180, 181],
                'distance': 2.3
            }
        ]

    @pytest.fixture
    def sample_residue_atom_mapping(self):
        """Sample residue to atom index mapping."""
        return {
            'ARG45.A': [120, 121, 122, 123, 124, 125, 126],
            'PHE89.A': [280, 281, 282, 283, 284, 285, 286, 287, 288, 289],
            'ASP67.A': [180, 181, 182, 183, 184, 185, 186, 187]
        }

    # ================================================================================================
    # UNIT TESTS - generate_ligand_heatmap
    # ================================================================================================

    @pytest.mark.unit
    @pytest.mark.requires_rdkit
    def test_generate_ligand_heatmap_basic(self, sample_ligand_energy_differences, water_files, temp_dir):
        """Test basic ligand heatmap generation."""
        ligand_path = water_files['xyz']
        output_path = temp_dir / "test_ligand_heatmap.png"

        # Generate heatmap
        fig = generate_ligand_heatmap(
            ligand_path=ligand_path,
            ligand_energy_differences=sample_ligand_energy_differences,
            output_path=output_path,
            title="Test Ligand Heatmap"
        )

        # Verify figure was created
        assert fig is not None
        assert isinstance(fig, plt.Figure)

        # Verify output file was created
        assert output_path.exists()

        # Verify figure has correct number of subplots (excluding colorbars)
        # Colorbar axes have xlabel with energy information
        main_axes = [ax for ax in fig.axes if not ax.get_xlabel() or ('total' not in ax.get_xlabel() and 'Energy' not in ax.get_xlabel())]
        assert len(main_axes) == len(sample_ligand_energy_differences)

        plt.close(fig)

    @pytest.mark.unit
    @pytest.mark.requires_rdkit
    def test_generate_ligand_heatmap_single_component(self, water_files, temp_dir):
        """Test ligand heatmap with single energy component."""
        ligand_path = water_files['xyz']
        single_component = {'MLFF': np.array([0.1, -0.2, 0.05])}

        fig = generate_ligand_heatmap(
            ligand_path=ligand_path,
            ligand_energy_differences=single_component
        )

        assert fig is not None
        # Should have 1 main plot + 1 colorbar = 2 total axes
        main_axes = [ax for ax in fig.axes if not ax.get_xlabel() or ('total' not in ax.get_xlabel() and 'Energy' not in ax.get_xlabel())]
        assert len(main_axes) == 1
        plt.close(fig)

    @pytest.mark.unit
    @pytest.mark.requires_rdkit
    def test_generate_ligand_heatmap_zero_weights(self, water_files):
        """Test ligand heatmap with zero energy differences."""
        ligand_path = water_files['xyz']
        zero_weights = {'MLFF': np.array([0.0, 0.0, 0.0])}

        fig = generate_ligand_heatmap(
            ligand_path=ligand_path,
            ligand_energy_differences=zero_weights
        )

        assert fig is not None
        plt.close(fig)

    @pytest.mark.unit
    @pytest.mark.requires_rdkit
    def test_generate_ligand_heatmap_no_output_path(self, sample_ligand_energy_differences, water_files):
        """Test ligand heatmap without saving to file."""
        ligand_path = water_files['xyz']

        fig = generate_ligand_heatmap(
            ligand_path=ligand_path,
            ligand_energy_differences=sample_ligand_energy_differences
        )

        assert fig is not None
        plt.close(fig)

    # ================================================================================================
    # UNIT TESTS - generate_protein_interaction_heatmap
    # ================================================================================================

    @pytest.mark.unit
    @pytest.mark.requires_rdkit
    def test_generate_protein_interaction_heatmap_basic(
        self,
        sample_ligand_energy_differences,
        sample_protein_energy_differences,
        sample_atom_mappings,
        sample_residue_atom_mapping,
        water_files,
        temp_dir
    ):
        """Test basic protein interaction heatmap generation."""
        ligand_path = water_files['xyz']

        fig = generate_protein_interaction_heatmap(
            ligand_path=ligand_path,
            ligand_energy_differences=sample_ligand_energy_differences,
            atom_mappings=sample_atom_mappings,
            protein_energy_differences=sample_protein_energy_differences,
            residue_atom_mapping=sample_residue_atom_mapping,
            output_path=temp_dir / "test_protein_heatmap.png",
            title="Test Protein Interaction Heatmap"
        )
        output_path = temp_dir / "test_protein_heatmap.png"

        assert fig is not None
        assert isinstance(fig, plt.Figure)
        assert output_path.exists()
        # Verify figure has correct number of subplots (excluding colorbars)
        main_axes = [ax for ax in fig.axes if not ax.get_xlabel() or ('total' not in ax.get_xlabel() and 'Energy' not in ax.get_xlabel())]
        assert len(main_axes) == len(sample_ligand_energy_differences)

        plt.close(fig)

    @pytest.mark.unit
    @pytest.mark.requires_rdkit
    def test_generate_protein_interaction_heatmap_no_protein_data(
        self,
        sample_ligand_energy_differences,
        sample_atom_mappings,
        water_files
    ):
        """Test protein interaction heatmap without protein energy data."""
        ligand_path = water_files['xyz']

        fig = generate_protein_interaction_heatmap(
            ligand_path=ligand_path,
            ligand_energy_differences=sample_ligand_energy_differences,
            atom_mappings=sample_atom_mappings
        )

        assert fig is not None
        plt.close(fig)

    @pytest.mark.unit
    @pytest.mark.requires_rdkit
    def test_generate_protein_interaction_heatmap_empty_mappings(
        self,
        sample_ligand_energy_differences,
        water_files
    ):
        """Test protein interaction heatmap with empty atom mappings."""
        ligand_path = water_files['xyz']

        fig = generate_protein_interaction_heatmap(
            ligand_path=ligand_path,
            ligand_energy_differences=sample_ligand_energy_differences,
            atom_mappings=[]
        )

        assert fig is not None
        plt.close(fig)

    @pytest.mark.unit
    @pytest.mark.requires_rdkit
    def test_generate_protein_interaction_heatmap_duplicate_residues(
        self,
        sample_ligand_energy_differences,
        water_files
    ):
        """Test protein interaction heatmap with duplicate residue interactions."""
        ligand_path = water_files['xyz']

        # Create mappings with same residue having multiple interactions
        duplicate_mappings = [
            {
                'frame': 0,
                'ligand_residue': 'UNL1',
                'protein_residue': 'ARG45.A',
                'interaction_type': 'HBDonor',
                'ligand_atoms': [0],
                'protein_atoms': [120, 121]
            },
            {
                'frame': 0,
                'ligand_residue': 'UNL1',
                'protein_residue': 'ARG45.A',
                'interaction_type': 'Electrostatic',
                'ligand_atoms': [1],
                'protein_atoms': [122, 123]
            }
        ]

        fig = generate_protein_interaction_heatmap(
            ligand_path=ligand_path,
            ligand_energy_differences=sample_ligand_energy_differences,
            atom_mappings=duplicate_mappings
        )

        assert fig is not None
        plt.close(fig)

    # ================================================================================================
    # UNIT TESTS - generate_energy_heatmap
    # ================================================================================================

    @pytest.mark.unit
    @pytest.mark.requires_rdkit
    def test_generate_energy_heatmap_ligand_only(self, sample_ligand_energy_differences, water_files):
        """Test energy heatmap generation for ligand only (no protein data)."""
        ligand_path = water_files['xyz']

        fig = generate_energy_heatmap(
            ligand_path=ligand_path,
            ligand_energy_differences=sample_ligand_energy_differences,
            output_paths=(None, None, None)
        )

        assert fig is not None
        plt.close(fig)

    @pytest.mark.unit
    @pytest.mark.requires_rdkit
    def test_generate_energy_heatmap_with_protein(
        self,
        sample_ligand_energy_differences,
        sample_protein_energy_differences,
        sample_residue_atom_mapping,
        water_files
    ):
        """Test energy heatmap generation with protein data."""
        ligand_path = water_files['xyz']

        # Mock preloaded_protein_prolif data
        mock_prolif_mol = Mock()
        mock_preloaded_protein = (mock_prolif_mol, sample_residue_atom_mapping)

        with patch('src.explainability._fp_interaction_mapping') as mock_fp_mapping:
            # Mock the interaction mapping function
            mock_fp_mapping.return_value = [
                {
                    'frame': 0,
                    'ligand_residue': 'UNL1',
                    'protein_residue': 'ARG45.A',
                    'interaction_type': 'HBDonor',
                    'ligand_atoms': [0],
                    'protein_atoms': [120]
                }
            ]

            fig = generate_energy_heatmap(
                ligand_path=ligand_path,
                ligand_energy_differences=sample_ligand_energy_differences,
                output_paths=(None, "/tmp/test_output.png", None),  # Provide output path for protein heatmap
                protein_energy_differences=sample_protein_energy_differences,
                preloaded_protein_prolif=mock_preloaded_protein
            )

            assert fig is not None
            assert mock_fp_mapping.called
            plt.close(fig)

    # ================================================================================================
    # UNIT TESTS - _fp_interaction_mapping
    # ================================================================================================

    @pytest.mark.unit
    @pytest.mark.requires_rdkit
    def test_fp_interaction_mapping(self):
        """Test fingerprint interaction mapping function."""
        # Mock ProLIF protein molecule - needs to support indexing with [0]
        mock_residue = Mock()
        mock_protein_prolif = Mock()
        mock_protein_prolif.__getitem__ = Mock(return_value=mock_residue)
        mock_ligand_mol = Mock()

        with patch('src.explainability.compute_protein_ligand_interactions') as mock_compute, \
             patch('src.explainability.get_atom_mappings') as mock_get_mappings:

            mock_fp = Mock()
            mock_compute.return_value = mock_fp
            mock_get_mappings.return_value = [{'test': 'mapping'}]

            result = _fp_interaction_mapping(mock_protein_prolif, mock_ligand_mol)

            assert result == [{'test': 'mapping'}]
            mock_compute.assert_called_once_with(mock_residue, mock_ligand_mol)
            mock_get_mappings.assert_called_once_with(mock_fp)

    # ================================================================================================
    # INTEGRATION TESTS - Real data
    # ================================================================================================

    @pytest.mark.integration
    @pytest.mark.requires_rdkit
    def test_ligand_heatmap_with_real_data(self, temp_dir):
        """Integration test with real molecular data."""
        # Use alanine as a more complex test molecule
        test_data_dir = Path(__file__).parent / "test_data"
        alanine_file = test_data_dir / "alanine.xyz"

        if not alanine_file.exists():
            pytest.skip("Alanine test data not available")

        # Generate realistic energy differences for alanine (13 atoms)
        ligand_energy_differences = {
            'MLFF': np.random.normal(0, 0.1, 13),
            'Electrostatics': np.random.normal(0, 0.05, 13),
            'Dispersion': np.random.normal(0, 0.02, 13)
        }
        ligand_energy_differences['Total'] = (
            ligand_energy_differences['MLFF'] +
            ligand_energy_differences['Electrostatics'] +
            ligand_energy_differences['Dispersion']
        )

        output_path = temp_dir / "alanine_heatmap_integration.png"

        fig = generate_ligand_heatmap(
            ligand_path=alanine_file,
            ligand_energy_differences=ligand_energy_differences,
            output_path=output_path,
            title="Alanine Integration Test"
        )

        assert fig is not None
        assert output_path.exists()
        assert output_path.stat().st_size > 0  # File has content
        # assert the number of figures
        main_axes = [ax for ax in fig.axes if not ax.get_xlabel() or ('total' not in ax.get_xlabel() and 'Energy' not in ax.get_xlabel())]
        assert len(main_axes) == len(ligand_energy_differences.keys())

        plt.close(fig)

    @pytest.mark.integration
    @pytest.mark.requires_rdkit
    def test_protein_heatmap_with_realistic_interactions(
        self,
        sample_protein_energy_differences,
        temp_dir
    ):
        """Integration test with realistic protein-ligand interactions."""
        test_data_dir = Path(__file__).parent / "test_data"
        water_file = test_data_dir / "water.xyz"

        if not water_file.exists():
            pytest.skip("Water test data not available")

        # Realistic water energy differences (3 atoms: O, H, H)
        water_energy_differences = {
            'MLFF': np.array([-0.15, 0.05, 0.05]),  # Oxygen more negative
            'Electrostatics': np.array([0.1, -0.05, -0.05]),  # Charge distribution
            'Dispersion': np.array([-0.01, 0.005, 0.005]),
            'Total': np.array([-0.06, 0.005, 0.005])
        }

        # Realistic protein-water interactions
        realistic_mappings = [
            {
                'frame': 0,
                'ligand_residue': 'WAT1',
                'protein_residue': 'SER123.A',
                'interaction_type': 'HBDonor',
                'ligand_atoms': [1],  # Hydrogen donor
                'protein_atoms': [450, 451]
            },
            {
                'frame': 0,
                'ligand_residue': 'WAT1',
                'protein_residue': 'ASP89.A',
                'interaction_type': 'HBAcceptor',
                'ligand_atoms': [0],  # Oxygen acceptor
                'protein_atoms': [320, 321]
            }
        ]

        # Realistic residue mapping
        residue_mapping = {
            'SER123.A': [450, 451, 452, 453, 454, 455],
            'ASP89.A': [320, 321, 322, 323, 324, 325, 326, 327]
        }


        fig = generate_protein_interaction_heatmap(
            ligand_path=water_file,
            ligand_energy_differences=water_energy_differences,
            atom_mappings=realistic_mappings,
            protein_energy_differences=sample_protein_energy_differences,
            residue_atom_mapping=residue_mapping,
            output_path=temp_dir / "water_protein_interaction.png",
            title="Water-Protein Interaction Integration Test"
        )
        output_path = temp_dir / "water_protein_interaction.png"
        assert fig is not None
        assert output_path.exists()
        assert output_path.stat().st_size > 0

        plt.close(fig)

    # ================================================================================================
    # ERROR HANDLING TESTS
    # ================================================================================================

    @pytest.mark.unit
    def test_generate_ligand_heatmap_invalid_path(self, sample_ligand_energy_differences):
        """Test ligand heatmap with invalid file path."""
        invalid_path = Path("/nonexistent/path/molecule.xyz")

        with pytest.raises((FileNotFoundError, OSError, ValueError)):
            generate_ligand_heatmap(
                ligand_path=invalid_path,
                ligand_energy_differences=sample_ligand_energy_differences
            )

    @pytest.mark.unit
    @pytest.mark.requires_rdkit
    def test_generate_ligand_heatmap_mismatched_atoms(self, water_files):
        """Test ligand heatmap with mismatched atom count."""
        ligand_path = water_files['xyz']
        # Water has 3 atoms, but provide weights for 5 atoms
        mismatched_weights = {'MLFF': np.array([0.1, 0.2, 0.3, 0.4, 0.5])}

        with pytest.raises(ValueError):
            generate_ligand_heatmap(
                ligand_path=ligand_path,
                ligand_energy_differences=mismatched_weights
            )

    @pytest.mark.unit
    def test_generate_ligand_heatmap_empty_weights(self, water_files):
        """Test ligand heatmap with empty energy differences."""
        ligand_path = water_files['xyz']

        with pytest.raises((ValueError, KeyError)):
            generate_ligand_heatmap(
                ligand_path=ligand_path,
                ligand_energy_differences={}
            )

    @pytest.mark.unit
    @pytest.mark.requires_rdkit
    def test_generate_protein_heatmap_invalid_atom_indices(
        self,
        sample_ligand_energy_differences,
        water_files
    ):
        """Test protein heatmap with invalid atom indices in mappings."""
        ligand_path = water_files['xyz']  # Water has atoms 0, 1, 2

        invalid_mappings = [
            {
                'frame': 0,
                'ligand_residue': 'UNL1',
                'protein_residue': 'ARG45.A',
                'interaction_type': 'HBDonor',
                'ligand_atoms': [10, 20],  # Invalid indices for water
                'protein_atoms': [120, 121]
            }
        ]

        # Should not raise error, but invalid indices should be ignored
        fig = generate_protein_interaction_heatmap(
            ligand_path=ligand_path,
            ligand_energy_differences=sample_ligand_energy_differences,
            atom_mappings=invalid_mappings
        )

        assert fig is not None
        plt.close(fig)