"""
Tests for the explain_utils module.
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from src.explain_utils import (
    create_colorbar,
    similarity_map_gen,
    categorize_interaction,
    get_interaction_color,
    add_interaction_summary,
    compute_protein_ligand_interactions,
    get_atom_mappings,
    residue_weights_calculation,
    compute_energy_differences
)


class TestExplainUtilsModule:
    """Tests for explain_utils functions."""

    @pytest.fixture
    def sample_figure(self):
        """Create a sample matplotlib figure for testing."""
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        yield fig, ax
        plt.close(fig)

    @pytest.fixture
    def sample_weights(self):
        """Sample weight array for testing."""
        return np.array([0.1, -0.2, 0.3, -0.1, 0.05])

    @pytest.fixture
    def sample_prolif_fingerprint(self):
        """Sample ProLIF fingerprint structure for testing."""
        # Mock the fingerprint structure that ProLIF returns
        mock_fp = Mock()
        mock_fp.ifp = {
            0: {  # frame 0
                ('UNL1', 'ARG45.A'): {
                    'HBDonor': [{'indices': {'ligand': [0, 1], 'protein': [120, 121]}, 'distance': 2.1}]
                },
                ('UNL1', 'PHE89.A'): {
                    'Hydrophobic': [{'indices': {'ligand': [2], 'protein': [280, 281]}, 'distance': 3.8}]
                }
            }
        }
        return mock_fp

    @pytest.fixture
    def sample_energy_components(self):
        """Sample energy components for testing."""
        return {
            'mlff_atomic_energy': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            'zbl_repulsion': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            'electrostatic_energy': np.array([0.05, 0.1, 0.15, 0.2, 0.25]),
            'dispersion_energy': np.array([-0.01, -0.02, -0.03, -0.04, -0.05])
        }

    # ================================================================================================
    # UNIT TESTS - Visualization utilities
    # ================================================================================================

    @pytest.mark.unit
    def test_create_colorbar_basic(self, sample_figure, sample_weights):
        """Test basic colorbar creation."""
        fig, ax = sample_figure
        global_max = 0.3
        component = "MLFF"

        create_colorbar(fig, ax, global_max, component, sample_weights)

        # Check that colorbar was added to figure
        assert len(fig.axes) >= 2  # Original ax + colorbar ax

    @pytest.mark.unit
    def test_create_colorbar_total_component(self, sample_figure, sample_weights):
        """Test colorbar creation for Total component."""
        fig, ax = sample_figure
        global_max = 0.5
        component = "Total"

        create_colorbar(fig, ax, global_max, component, sample_weights)

        # Should handle Total component differently in label
        assert len(fig.axes) >= 2

    @pytest.mark.unit
    def test_create_colorbar_zero_weights(self, sample_figure):
        """Test colorbar creation with zero weights."""
        fig, ax = sample_figure
        global_max = 1e-6  # Very small value
        component = "MLFF"
        zero_weights = np.array([0.0, 0.0, 0.0])

        create_colorbar(fig, ax, global_max, component, zero_weights)

        assert len(fig.axes) >= 2

    @pytest.mark.unit
    @pytest.mark.requires_rdkit
    def test_similarity_map_gen_basic(self, water_files):
        """Test basic similarity map generation."""
        from src.molecule_loader import load_molecule_to_prolif

        mol = load_molecule_to_prolif(water_files['xyz'])
        rdkit_mol = mol.mol if hasattr(mol, 'mol') else mol

        weights = [0.1, -0.2, 0.05]  # 3 atoms for water

        img = similarity_map_gen(
            mol=rdkit_mol,
            weights=weights,
            cmap="bwr",
            width=300,
            height=300
        )

        assert img is not None
        assert isinstance(img, np.ndarray)
        assert img.shape[2] in [3, 4]  # RGB or RGBA

    @pytest.mark.unit
    @pytest.mark.requires_rdkit
    def test_similarity_map_gen_mismatched_weights(self, water_files):
        """Test similarity map generation with mismatched weight count."""
        from src.molecule_loader import load_molecule_to_prolif

        mol = load_molecule_to_prolif(water_files['xyz'])
        rdkit_mol = mol.mol if hasattr(mol, 'mol') else mol

        # Water has 3 atoms, provide 5 weights
        weights = [0.1, -0.2, 0.05, 0.1, -0.1]

        with pytest.raises(ValueError, match="Weights length .* does not match number of atoms"):
            similarity_map_gen(
                mol=rdkit_mol,
                weights=weights
            )

    @pytest.mark.unit
    @pytest.mark.requires_rdkit
    def test_similarity_map_with_nan_weights(self, water_files):
        """Test similarity map with NaN weights."""
        from src.molecule_loader import load_molecule_to_prolif

        mol = load_molecule_to_prolif(water_files['xyz'])
        rdkit_mol = mol.mol if hasattr(mol, 'mol') else mol

        weights = [0.1, np.nan, 0.05]  # NaN weight for middle atom

        img = similarity_map_gen(
            mol=rdkit_mol,
            weights=weights
        )

        assert img is not None

    # ================================================================================================
    # UNIT TESTS - Interaction categorization and coloring
    # ================================================================================================

    @pytest.mark.unit
    def test_categorize_interaction_hbond_types(self):
        """Test categorization of hydrogen bond types."""
        assert categorize_interaction('HBDonor') == 'HBDonor'
        assert categorize_interaction('HBAcceptor') == 'HBAcceptor'
        assert categorize_interaction('hbond') == 'H-bond'
        assert categorize_interaction('H-bond') == 'H-bond'
        assert categorize_interaction('hydrophilic') == 'H-bond'

    @pytest.mark.unit
    def test_categorize_interaction_halogen_bonds(self):
        """Test categorization of halogen bond types."""
        assert categorize_interaction('XBDonor') == 'Halogen bond'
        assert categorize_interaction('XBAcceptor') == 'Halogen bond'
        assert categorize_interaction('halogen') == 'Halogen bond'

    @pytest.mark.unit
    def test_categorize_interaction_hydrophobic_vdw(self):
        """Test categorization of hydrophobic and VdW interactions."""
        assert categorize_interaction('Hydrophobic') == 'Hydrophobic'
        assert categorize_interaction('VdWContact') == 'VdW'
        assert categorize_interaction('vdw') == 'VdW'
        assert categorize_interaction('vanderwaals') == 'VdW'

    @pytest.mark.unit
    def test_categorize_interaction_pi_interactions(self):
        """Test categorization of pi interactions."""
        assert categorize_interaction('PiStacking') == 'π-interaction'
        assert categorize_interaction('PiCation') == 'π-interaction'
        assert categorize_interaction('CationPi') == 'π-interaction'
        assert categorize_interaction('EdgeToFace') == 'π-interaction'
        assert categorize_interaction('FaceToFace') == 'π-interaction'
        assert categorize_interaction('pi-stacking') == 'π-interaction'

    @pytest.mark.unit
    def test_categorize_interaction_ionic(self):
        """Test categorization of ionic interactions."""
        assert categorize_interaction('Anionic') == 'Ionic'
        assert categorize_interaction('Cationic') == 'Ionic'
        assert categorize_interaction('SaltBridge') == 'Ionic'
        assert categorize_interaction('salt-bridge') == 'Ionic'

    @pytest.mark.unit
    def test_categorize_interaction_metal(self):
        """Test categorization of metal interactions."""
        assert categorize_interaction('MetalDonor') == 'Metal'
        assert categorize_interaction('MetalAcceptor') == 'Metal'
        assert categorize_interaction('metal') == 'Metal'

    @pytest.mark.unit
    def test_categorize_interaction_distance_angle(self):
        """Test categorization of distance/angle measurements."""
        assert categorize_interaction('distance') == 'Distance/Angle'
        assert categorize_interaction('SingleAngle') == 'Distance/Angle'
        assert categorize_interaction('DoubleAngle') == 'Distance/Angle'

    @pytest.mark.unit
    def test_categorize_interaction_unknown(self):
        """Test categorization of unknown interaction types."""
        assert categorize_interaction('UnknownType') == 'Other'
        assert categorize_interaction('') == 'Other'
        assert categorize_interaction('RandomString') == 'Other'

    @pytest.mark.unit
    def test_get_interaction_color_hbond_types(self):
        """Test color assignment for hydrogen bond types."""
        # HBDonor should be light blue
        color = get_interaction_color('HBDonor')
        assert color == (0.4, 0.6, 1.0)

        # HBAcceptor should be medium blue
        color = get_interaction_color('HBAcceptor')
        assert color == (0.2, 0.4, 0.9)

        # General H-bonds should be dark blue
        color = get_interaction_color('hbond')
        assert color == (0.1, 0.3, 0.7)

    @pytest.mark.unit
    def test_get_interaction_color_hydrophobic_vdw(self):
        """Test color assignment for hydrophobic and VdW interactions."""
        # Hydrophobic should be dark green
        color = get_interaction_color('Hydrophobic')
        assert color == (0.1, 0.6, 0.1)

        # VdW should be light green
        color = get_interaction_color('VdWContact')
        assert color == (0.5, 0.9, 0.5)

    @pytest.mark.unit
    def test_get_interaction_color_pi_interactions(self):
        """Test color assignment for pi interactions."""
        # All pi interactions should be yellow
        for interaction in ['PiStacking', 'PiCation', 'EdgeToFace']:
            color = get_interaction_color(interaction)
            assert color == (0.9, 0.9, 0.2)

    @pytest.mark.unit
    def test_get_interaction_color_ionic_metal(self):
        """Test color assignment for ionic and metal interactions."""
        # Ionic should be cyan
        color = get_interaction_color('Anionic')
        assert color == (0.3, 0.8, 0.8)

        # Metal should be purple/magenta
        color = get_interaction_color('MetalDonor')
        assert color == (0.8, 0.2, 0.8)

    @pytest.mark.unit
    def test_get_interaction_color_unknown(self):
        """Test color assignment for unknown interactions."""
        color = get_interaction_color('UnknownType')
        assert color == (0.5, 0.5, 0.5)  # Default gray

    # ================================================================================================
    # UNIT TESTS - Interaction summary
    # ================================================================================================

    @pytest.mark.unit
    def test_add_interaction_summary_basic(self):
        """Test basic interaction summary addition."""
        fig = plt.figure(figsize=(10, 6))

        sample_mappings = [
            {'interaction_type': 'HBDonor'},
            {'interaction_type': 'HBDonor'},
            {'interaction_type': 'Hydrophobic'},
            {'interaction_type': 'PiStacking'}
        ]

        add_interaction_summary(fig, sample_mappings)

        # Should not raise any errors
        assert fig is not None
        plt.close(fig)

    @pytest.mark.unit
    def test_add_interaction_summary_empty_mappings(self):
        """Test interaction summary with empty mappings."""
        fig = plt.figure(figsize=(10, 6))

        add_interaction_summary(fig, [])

        # Should handle empty list gracefully
        assert fig is not None
        plt.close(fig)

    @pytest.mark.unit
    def test_add_interaction_summary_duplicate_types(self):
        """Test interaction summary with duplicate interaction types."""
        fig = plt.figure(figsize=(10, 6))

        sample_mappings = [
            {'interaction_type': 'HBDonor'},
            {'interaction_type': 'HBDonor'},
            {'interaction_type': 'HBDonor'},
            {'interaction_type': 'Hydrophobic'},
            {'interaction_type': 'Hydrophobic'}
        ]

        add_interaction_summary(fig, sample_mappings)

        assert fig is not None
        plt.close(fig)

    # ================================================================================================
    # UNIT TESTS - ProLIF interaction computation
    # ================================================================================================

    @pytest.mark.unit
    @pytest.mark.requires_rdkit
    def test_compute_protein_ligand_interactions(self):
        """Test protein-ligand interaction computation."""
        # Mock ProLIF molecules
        mock_protein = Mock()
        mock_ligand = Mock()

        with patch('prolif.Fingerprint') as mock_fp_class:
            mock_fp_instance = Mock()
            mock_fp_class.return_value = mock_fp_instance

            result = compute_protein_ligand_interactions(mock_protein, mock_ligand)

            # Should create fingerprint and run computation
            mock_fp_class.assert_called_once()
            mock_fp_instance.run_from_iterable.assert_called_once_with([mock_ligand], mock_protein)
            assert result == mock_fp_instance

    @pytest.mark.unit
    def test_get_atom_mappings_basic(self, sample_prolif_fingerprint):
        """Test extraction of atom mappings from ProLIF fingerprint."""
        mappings = get_atom_mappings(sample_prolif_fingerprint)

        assert len(mappings) == 2  # Two interactions in mock data

        # Check first mapping
        first_mapping = mappings[0]
        assert first_mapping['frame'] == 0
        assert first_mapping['ligand_residue'] == 'UNL1'
        assert first_mapping['protein_residue'] == 'ARG45.A'
        assert first_mapping['interaction_type'] == 'HBDonor'
        assert first_mapping['ligand_atoms'] == [0, 1]
        assert first_mapping['protein_atoms'] == [120, 121]
        assert first_mapping['distance'] == 2.1

        # Check second mapping
        second_mapping = mappings[1]
        assert second_mapping['interaction_type'] == 'Hydrophobic'
        assert second_mapping['protein_residue'] == 'PHE89.A'

    @pytest.mark.unit
    def test_get_atom_mappings_empty_fingerprint(self):
        """Test atom mapping extraction from empty fingerprint."""
        mock_fp = Mock()
        mock_fp.ifp = {}

        mappings = get_atom_mappings(mock_fp)

        assert mappings == []

    @pytest.mark.unit
    def test_get_atom_mappings_no_metadata(self):
        """Test atom mapping extraction with missing metadata."""
        mock_fp = Mock()
        mock_fp.ifp = {
            0: {
                ('UNL1', 'ARG45.A'): {
                    'HBDonor': [None]  # No metadata
                }
            }
        }

        mappings = get_atom_mappings(mock_fp)

        assert len(mappings) == 1
        mapping = mappings[0]
        assert mapping['frame'] == 0
        assert mapping['ligand_residue'] == 'UNL1'
        assert mapping['protein_residue'] == 'ARG45.A'
        assert mapping['interaction_type'] == 'HBDonor'
        # Should not have ligand_atoms, protein_atoms, or distance

    # ================================================================================================
    # UNIT TESTS - Residue weight calculation
    # ================================================================================================

    @pytest.mark.unit
    def test_residue_weights_calculation_basic(self):
        """Test basic residue weight calculation."""
        residue_atom_indices = {
            'ARG45.A': [120, 121, 122],
            'PHE89.A': [280, 281, 282, 283]
        }

        protein_energy_differences = {
            'MLFF': np.array([0.0] * 300),  # 300 atoms
            'Electrostatics': np.array([0.0] * 300)
        }

        # Set specific values for residue atoms
        protein_energy_differences['MLFF'][120:123] = [0.1, 0.2, 0.1]  # ARG45.A
        protein_energy_differences['MLFF'][280:284] = [0.05, 0.1, 0.15, 0.05]  # PHE89.A
        protein_energy_differences['Electrostatics'][120:123] = [0.02, 0.03, 0.01]
        protein_energy_differences['Electrostatics'][280:284] = [0.01, 0.02, 0.03, 0.01]

        result = residue_weights_calculation(residue_atom_indices, protein_energy_differences)

        assert 'ARG45.A' in result
        assert 'PHE89.A' in result

        # Check ARG45.A weights (sum of atoms 120, 121, 122)
        arg_weights = result['ARG45.A']
        assert arg_weights['MLFF'] == pytest.approx(0.4, abs=1e-6)  # 0.1 + 0.2 + 0.1
        assert arg_weights['Electrostatics'] == pytest.approx(0.06, abs=1e-6)  # 0.02 + 0.03 + 0.01
        assert arg_weights['Total'] == pytest.approx(0.46, abs=1e-6)  # 0.4 + 0.06

        # Check PHE89.A weights
        phe_weights = result['PHE89.A']
        assert phe_weights['MLFF'] == pytest.approx(0.35, abs=1e-6)  # 0.05 + 0.1 + 0.15 + 0.05
        assert phe_weights['Electrostatics'] == pytest.approx(0.07, abs=1e-6)  # 0.01 + 0.02 + 0.03 + 0.01

    @pytest.mark.unit
    def test_residue_weights_calculation_out_of_bounds(self):
        """Test residue weight calculation with out-of-bounds indices."""
        residue_atom_indices = {
            'TEST_RES': [100, 150, 200]  # Only index 100 is within bounds
        }

        protein_energy_differences = {
            'MLFF': np.array([0.1] * 120),  # Only 120 atoms
            'Electrostatics': np.array([0.05] * 120)
        }

        result = residue_weights_calculation(residue_atom_indices, protein_energy_differences)

        assert 'TEST_RES' in result
        # Should only include the valid index (100)
        assert result['TEST_RES']['MLFF'] == pytest.approx(0.1, abs=1e-6)
        assert result['TEST_RES']['Electrostatics'] == pytest.approx(0.05, abs=1e-6)

    @pytest.mark.unit
    def test_residue_weights_calculation_empty_residues(self):
        """Test residue weight calculation with empty residue list."""
        residue_atom_indices = {}
        protein_energy_differences = {
            'MLFF': np.array([0.1, 0.2, 0.3]),
            'Electrostatics': np.array([0.01, 0.02, 0.03])
        }

        result = residue_weights_calculation(residue_atom_indices, protein_energy_differences)

        assert result == {}

    # ================================================================================================
    # UNIT TESTS - Energy difference computation
    # ================================================================================================

    @pytest.mark.unit
    def test_compute_energy_differences_basic(self, sample_energy_components):
        """Test basic energy difference computation."""
        # Create complex components (protein + ligand)
        n_protein_atoms = 5
        n_ligand_atoms = 5

        protein_components = sample_energy_components.copy()
        ligand_components = sample_energy_components.copy()

        # Complex = protein + ligand concatenated
        complex_components = {}
        for key in sample_energy_components:
            complex_components[key] = np.concatenate([
                protein_components[key],  # First 5 atoms (protein)
                ligand_components[key] + 0.1  # Next 5 atoms (ligand, slightly different)
            ])

        ligand_diff, protein_diff = compute_energy_differences(
            protein_components=protein_components,
            ligand_components=ligand_components,
            complex_components=complex_components,
            n_protein_atoms=n_protein_atoms,
            n_ligand_atoms=n_ligand_atoms,
            protein_mode=True
        )

        # Check ligand differences
        assert 'MLFF' in ligand_diff
        assert 'Electrostatics' in ligand_diff
        assert 'Dispersion' in ligand_diff
        assert 'Total' in ligand_diff

        # Ligand differences should be complex_ligand - ligand_alone = +0.1 for each component
        expected_ligand_diff = [0.1] * 5
        assert np.allclose(ligand_diff['MLFF'], expected_ligand_diff)

        # Check protein differences
        assert protein_diff is not None
        assert 'MLFF' in protein_diff
        # Protein differences should be complex_protein - protein_alone = 0.0
        expected_protein_diff = [0.0] * 5
        assert np.allclose(protein_diff['MLFF'], expected_protein_diff)

    @pytest.mark.unit
    def test_compute_energy_differences_ligand_only(self, sample_energy_components):
        """Test energy difference computation in ligand-only mode."""
        n_protein_atoms = 3
        n_ligand_atoms = 3

        protein_components = {
            'mlff_atomic_energy': np.array([0.1, 0.2, 0.3]),
            'electrostatic_energy': np.array([0.01, 0.02, 0.03])
        }

        ligand_components = {
            'mlff_atomic_energy': np.array([0.05, 0.1, 0.15]),
            'electrostatic_energy': np.array([0.005, 0.01, 0.015])
        }

        complex_components = {
            'mlff_atomic_energy': np.array([0.1, 0.2, 0.3, 0.06, 0.11, 0.16]),  # protein + ligand
            'electrostatic_energy': np.array([0.01, 0.02, 0.03, 0.006, 0.011, 0.016])
        }

        ligand_diff, protein_diff = compute_energy_differences(
            protein_components=protein_components,
            ligand_components=ligand_components,
            complex_components=complex_components,
            n_protein_atoms=n_protein_atoms,
            n_ligand_atoms=n_ligand_atoms,
            protein_mode=False
        )

        # Should return ligand differences only
        assert ligand_diff is not None
        assert protein_diff is None

        # Check ligand differences: complex_ligand - ligand_alone
        expected_mlff = [0.01, 0.01, 0.01]  # [0.06, 0.11, 0.16] - [0.05, 0.1, 0.15]
        assert np.allclose(ligand_diff['MLFF'], expected_mlff)

    @pytest.mark.unit
    def test_compute_energy_differences_missing_components(self):
        """Test energy difference computation with missing components."""
        protein_components = {'mlff_atomic_energy': np.array([0.1, 0.2])}
        ligand_components = {'mlff_atomic_energy': np.array([0.05, 0.1])}
        complex_components = {'different_component': np.array([0.1, 0.2, 0.05, 0.1])}

        ligand_diff, protein_diff = compute_energy_differences(
            protein_components=protein_components,
            ligand_components=ligand_components,
            complex_components=complex_components,
            n_protein_atoms=2,
            n_ligand_atoms=2,
            protein_mode=True
        )

        # Should return empty dictionaries when no matching components found
        assert ligand_diff == {}
        assert protein_diff is None

    # ================================================================================================
    # INTEGRATION TESTS
    # ================================================================================================

    @pytest.mark.integration
    @pytest.mark.requires_rdkit
    def test_full_workflow_with_real_molecules(self, water_files, alanine_files):
        """Integration test combining multiple explain_utils functions."""
        from src.molecule_loader import load_molecule_to_prolif

        # Skip if required files not available
        if not water_files or not alanine_files:
            pytest.skip("Required test molecules not available")

        # Load real molecules
        water_mol = load_molecule_to_prolif(water_files['xyz'])
        water_rdkit = water_mol.mol if hasattr(water_mol, 'mol') else water_mol

        # Test similarity map generation
        water_weights = [0.1, -0.05, 0.02]  # 3 atoms for water
        img = similarity_map_gen(water_rdkit, water_weights)
        assert img is not None

        # Test interaction categorization
        interactions = ['HBDonor', 'Hydrophobic', 'PiStacking', 'UnknownType']
        categories = [categorize_interaction(i) for i in interactions]
        expected = ['HBDonor', 'Hydrophobic', 'π-interaction', 'Other']
        assert categories == expected

        # Test color assignment
        colors = [get_interaction_color(i) for i in interactions]
        assert len(colors) == 4
        assert all(len(color) == 3 for color in colors)  # RGB tuples

    @pytest.mark.integration
    def test_energy_difference_computation_realistic(self):
        """Integration test with realistic energy difference computation."""
        # Simulate realistic protein-ligand system
        n_protein = 100
        n_ligand = 20

        # Generate realistic energy components
        np.random.seed(42)  # For reproducible tests

        protein_alone = {
            'mlff_atomic_energy': np.random.normal(-5.0, 1.0, n_protein),
            'electrostatic_energy': np.random.normal(0.0, 0.5, n_protein),
            'dispersion_energy': np.random.normal(-0.1, 0.05, n_protein)
        }

        ligand_alone = {
            'mlff_atomic_energy': np.random.normal(-2.0, 0.5, n_ligand),
            'electrostatic_energy': np.random.normal(0.0, 0.2, n_ligand),
            'dispersion_energy': np.random.normal(-0.05, 0.02, n_ligand)
        }

        # Complex: slight perturbation due to interaction
        complex_protein = {k: v + np.random.normal(0, 0.01, n_protein) for k, v in protein_alone.items()}
        complex_ligand = {k: v + np.random.normal(0, 0.02, n_ligand) for k, v in ligand_alone.items()}

        complex_combined = {}
        for key in protein_alone:
            complex_combined[key] = np.concatenate([complex_protein[key], complex_ligand[key]])

        ligand_diff, protein_diff = compute_energy_differences(
            protein_components=protein_alone,
            ligand_components=ligand_alone,
            complex_components=complex_combined,
            n_protein_atoms=n_protein,
            n_ligand_atoms=n_ligand,
            protein_mode=True
        )

        # Check structure of results
        assert ligand_diff is not None
        assert protein_diff is not None

        for component in ['MLFF', 'Electrostatics', 'Dispersion', 'Total']:
            assert component in ligand_diff
            assert component in protein_diff
            assert len(ligand_diff[component]) == n_ligand
            assert len(protein_diff[component]) == n_protein

        # Energy differences should be small (perturbations)
        for component in ['MLFF', 'Electrostatics', 'Dispersion']:
            assert np.max(np.abs(ligand_diff[component])) < 0.1  # Small changes
            assert np.max(np.abs(protein_diff[component])) < 0.1

    # ================================================================================================
    # ERROR HANDLING TESTS
    # ================================================================================================

    @pytest.mark.unit
    def test_residue_weights_calculation_invalid_energy_shape(self):
        """Test residue weight calculation with invalid energy array shapes."""
        residue_atom_indices = {'RES1': [0, 1, 2]}

        # Mismatched energy component lengths
        protein_energy_differences = {
            'MLFF': np.array([0.1, 0.2]),  # 2 atoms
            'Electrostatics': np.array([0.01, 0.02, 0.03])  # 3 atoms
        }

        # Should handle gracefully - use shortest array
        result = residue_weights_calculation(residue_atom_indices, protein_energy_differences)

        assert 'RES1' in result
        # Only indices 0, 1 should contribute (index 2 out of bounds for MLFF)
        assert result['RES1']['MLFF'] == pytest.approx(0.3, abs=1e-6)  # 0.1 + 0.2

    @pytest.mark.unit
    def test_compute_energy_differences_size_mismatch(self):
        """Test energy difference computation with size mismatches."""
        protein_components = {'mlff_atomic_energy': np.array([0.1, 0.2])}
        ligand_components = {'mlff_atomic_energy': np.array([0.05])}
        complex_components = {'mlff_atomic_energy': np.array([0.1, 0.2, 0.06, 0.11])}  # 4 total

        # n_protein=2, n_ligand=1, but complex has 4 atoms
        ligand_diff, protein_diff = compute_energy_differences(
            protein_components=protein_components,
            ligand_components=ligand_components,
            complex_components=complex_components,
            n_protein_atoms=2,
            n_ligand_atoms=1,
            protein_mode=True
        )

        # Should handle the mismatch gracefully
        assert ligand_diff is not None
        assert protein_diff is not None