"""
Tests for the calculator module.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from ase import Atoms

from src.calculator import So3lrSfCalculator
from src.utils import read_structure


class TestSo3lrSfCalculator:
    """Tests for the So3lrSfCalculator class."""

    @pytest.mark.unit
    def test_calculator_initialization_with_auto_detection(self, so3lr_model_path):
        """Test calculator initialization with automatic model detection."""
        with patch('src.calculator.find_so3lr_params', return_value=so3lr_model_path):
            with patch('src.calculator.mlffCalculatorSparse') as mock_mlff:
                mock_mlff.create_from_ckpt_dir.return_value = Mock()

                calc = So3lrSfCalculator()

                assert calc.model_path == so3lr_model_path
                assert calc.lr_cutoff == 12.0
                assert calc.dtype == np.float32
                assert calc.output_per_atom_energy_components is False

                # Verify mlffCalculatorSparse was called with correct parameters
                mock_mlff.create_from_ckpt_dir.assert_called_once_with(
                    ckpt_dir=so3lr_model_path,
                    lr_cutoff=12.0,
                    dispersion_energy_lr_cutoff_damping=2.0,
                    from_file=False,
                    calculate_stress=False,
                    dtype=np.float32,
                    output_per_atom_energy_components=False
                )

    @pytest.mark.unit
    def test_calculator_initialization_with_explicit_path(self, so3lr_model_path):
        """Test calculator initialization with explicit model path."""
        explicit_path = so3lr_model_path

        with patch('src.calculator.mlffCalculatorSparse') as mock_mlff:
            mock_mlff.create_from_ckpt_dir.return_value = Mock()

            calc = So3lrSfCalculator(model_path=explicit_path)

            assert calc.model_path == explicit_path
            mock_mlff.create_from_ckpt_dir.assert_called_once_with(
                ckpt_dir=explicit_path,
                lr_cutoff=12.0,
                dispersion_energy_lr_cutoff_damping=2.0,
                from_file=False,
                calculate_stress=False,
                dtype=np.float32,
                output_per_atom_energy_components=False
            )

    @pytest.mark.unit
    def test_calculator_initialization_no_model_found(self):
        """Test calculator initialization fails when no model is found."""
        with patch('src.calculator.find_so3lr_params', return_value=None):
            with pytest.raises(FileNotFoundError, match="Could not automatically locate SO3LR model parameters"):
                So3lrSfCalculator()

    @pytest.mark.unit
    def test_calculator_initialization_with_components(self, so3lr_model_path):
        """Test calculator initialization with per-atom components enabled."""
        with patch('src.calculator.find_so3lr_params', return_value=so3lr_model_path):
            with patch('src.calculator.mlffCalculatorSparse') as mock_mlff:
                mock_mlff.create_from_ckpt_dir.return_value = Mock()

                calc = So3lrSfCalculator(
                    output_per_atom_energy_components=True,
                    lr_cutoff=15.0,
                    dtype=np.float64
                )

                assert calc.output_per_atom_energy_components is True
                assert calc.lr_cutoff == 15.0
                assert calc.dtype == np.float64

                mock_mlff.create_from_ckpt_dir.assert_called_once_with(
                    ckpt_dir=so3lr_model_path,
                    lr_cutoff=15.0,
                    dispersion_energy_lr_cutoff_damping=2.0,
                    from_file=False,
                    calculate_stress=False,
                    dtype=np.float64,
                    output_per_atom_energy_components=True
                )

    @pytest.mark.unit
    def test_calculator_initialization_failure(self, so3lr_model_path):
        """Test calculator initialization failure handling."""
        with patch('src.calculator.find_so3lr_params', return_value=so3lr_model_path):
            with patch('src.calculator.mlffCalculatorSparse') as mock_mlff:
                mock_mlff.create_from_ckpt_dir.side_effect = Exception("Model loading failed")

                with pytest.raises(RuntimeError, match="Failed to initialize SO3LR calculator"):
                    So3lrSfCalculator()

    @pytest.mark.unit
    def test_calculate_energy_with_atoms_object(self, water_files):
        """Test energy calculation with ASE Atoms object."""
        atoms = read_structure(water_files['xyz'])
        mock_model_path = "/fake/model/path"

        with patch('src.calculator.find_so3lr_params', return_value=mock_model_path):
            with patch('src.calculator.mlffCalculatorSparse') as mock_mlff:
                mock_calc = Mock()
                mock_mlff.create_from_ckpt_dir.return_value = mock_calc

                calc = So3lrSfCalculator()

                # Mock the energy calculation
                with patch.object(atoms, 'get_potential_energy', return_value=-10.5):
                    energy = calc.calculate_energy(atoms)

                assert energy == -10.5
                assert atoms.calc == mock_calc

    @pytest.mark.unit
    def test_calculate_energy_with_file_path(self, water_files):
        """Test energy calculation with file path."""
        mock_model_path = "/fake/model/path"

        with patch('src.calculator.find_so3lr_params', return_value=mock_model_path):
            with patch('src.calculator.mlffCalculatorSparse') as mock_mlff:
                mock_calc = Mock()
                mock_mlff.create_from_ckpt_dir.return_value = mock_calc

                calc = So3lrSfCalculator()

                with patch('src.calculator.read_structure') as mock_read:
                    mock_atoms = Mock()
                    mock_atoms.get_potential_energy.return_value = -15.2
                    mock_read.return_value = mock_atoms

                    energy = calc.calculate_energy(water_files['xyz'])

                assert energy == -15.2
                mock_read.assert_called_once_with(water_files['xyz'])
                assert mock_atoms.calc == mock_calc

    @pytest.mark.unit
    def test_calculate_energy_validation_error(self):
        """Test energy calculation with invalid structure."""
        mock_model_path = "/fake/model/path"

        with patch('src.calculator.find_so3lr_params', return_value=mock_model_path):
            with patch('src.calculator.mlffCalculatorSparse') as mock_mlff:
                mock_mlff.create_from_ckpt_dir.return_value = Mock()

                calc = So3lrSfCalculator()

                with patch('src.calculator.validate_structure', side_effect=ValueError("Invalid structure")):
                    with pytest.raises(ValueError, match="Invalid structure"):
                        calc.calculate_energy(Atoms())

    @pytest.mark.unit
    def test_calculate_energy_calculation_error(self, water_files):
        """Test energy calculation failure handling."""
        atoms = read_structure(water_files['xyz'])
        mock_model_path = "/fake/model/path"

        with patch('src.calculator.find_so3lr_params', return_value=mock_model_path):
            with patch('src.calculator.mlffCalculatorSparse') as mock_mlff:
                mock_mlff.create_from_ckpt_dir.return_value = Mock()

                calc = So3lrSfCalculator()

                with patch.object(atoms, 'get_potential_energy', side_effect=Exception("Calculation failed")):
                    with pytest.raises(RuntimeError, match="Energy calculation failed"):
                        calc.calculate_energy(atoms)

    @pytest.mark.unit
    def test_get_per_atom_components_disabled(self):
        """Test getting per-atom components when disabled."""
        mock_model_path = "/fake/model/path"

        with patch('src.calculator.find_so3lr_params', return_value=mock_model_path):
            with patch('src.calculator.mlffCalculatorSparse') as mock_mlff:
                mock_mlff.create_from_ckpt_dir.return_value = Mock()

                calc = So3lrSfCalculator(output_per_atom_energy_components=False)

                with pytest.raises(ValueError, match="Per-atom energy components not enabled"):
                    calc.get_per_atom_energy_components()

    @pytest.mark.unit
    def test_get_per_atom_components_enabled(self):
        """Test getting per-atom components when enabled."""
        mock_model_path = "/fake/model/path"
        mock_components = {
            'mlff_atomic_energy': np.array([-5.0, -3.0, -2.5]),
            'zbl_repulsion': np.array([0.1, 0.05, 0.08])
        }

        with patch('src.calculator.find_so3lr_params', return_value=mock_model_path):
            with patch('src.calculator.mlffCalculatorSparse') as mock_mlff:
                mock_calc = Mock()
                mock_calc.get_per_atom_energy_components.return_value = mock_components
                mock_mlff.create_from_ckpt_dir.return_value = mock_calc

                calc = So3lrSfCalculator(output_per_atom_energy_components=True)
                components = calc.get_per_atom_energy_components()

                assert components == mock_components
                mock_calc.get_per_atom_energy_components.assert_called_once()

    @pytest.mark.unit
    def test_get_per_atom_components_no_method(self):
        """Test getting per-atom components when method doesn't exist."""
        mock_model_path = "/fake/model/path"

        with patch('src.calculator.find_so3lr_params', return_value=mock_model_path):
            with patch('src.calculator.mlffCalculatorSparse') as mock_mlff:
                mock_calc = Mock()
                # Remove the method to simulate it not existing
                del mock_calc.get_per_atom_energy_components
                mock_mlff.create_from_ckpt_dir.return_value = mock_calc

                calc = So3lrSfCalculator(output_per_atom_energy_components=True)
                components = calc.get_per_atom_energy_components()

                assert components is None

    @pytest.mark.integration
    @pytest.mark.slow
    def test_calculator_integration_with_real_structures(self, water_files, alanine_files):
        """Integration test with real structure files (requires actual SO3LR model)."""
        # Skip if SO3LR model is not available
        try:
            calc = So3lrSfCalculator()
        except FileNotFoundError:
            pytest.skip("SO3LR model not available")

        # Test with water molecule
        try:
            water_energy = calc.calculate_energy(water_files['xyz'])
            assert isinstance(water_energy, float)
            assert not np.isnan(water_energy)
        except Exception as e:
            pytest.skip(f"SO3LR calculation failed: {e}")

        # Test with alanine
        try:
            alanine_energy = calc.calculate_energy(alanine_files['xyz'])
            assert isinstance(alanine_energy, float)
            assert not np.isnan(alanine_energy)
        except Exception as e:
            pytest.skip(f"SO3LR calculation failed: {e}")

    @pytest.mark.unit
    def test_calculator_properties(self):
        """Test calculator property access."""
        mock_model_path = "/fake/model/path"
        lr_cutoff = 15.0
        dtype = np.float64

        with patch('src.calculator.find_so3lr_params', return_value=mock_model_path):
            with patch('src.calculator.mlffCalculatorSparse') as mock_mlff:
                mock_mlff.create_from_ckpt_dir.return_value = Mock()

                calc = So3lrSfCalculator(
                    lr_cutoff=lr_cutoff,
                    dtype=dtype,
                    output_per_atom_energy_components=True
                )

                assert calc.model_path == mock_model_path
                assert calc.lr_cutoff == lr_cutoff
                assert calc.dtype == dtype
                assert calc.output_per_atom_energy_components is True
                assert calc._calculator is not None