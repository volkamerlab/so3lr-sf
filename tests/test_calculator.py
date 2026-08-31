"""
Tests for the calculator module.
"""

import pytest
import numpy as np
from src.molecule_loader import load_ase_structure
from src.calculator import So3lrSfCalculator

class TestSo3lrSfCalculator:
    """Tests for the So3lrSfCalculator class."""

    # ================================================================================================
    # UNIT TESTS - MOCK CALCULATOR
    # ================================================================================================
    # The following tests use MockSo3lrSfCalculator to test functionality without requiring
    # actual SO3LR model files. These tests are fast and predictable:
    # - Mock calculator returns -100.0 for all energy calculations
    # - Mock components return predefined arrays for testing
    # - All initialization tests pass (no real model loading)
    # ================================================================================================

    @pytest.mark.unit
    def test_calculator_initialization_basic(self, mock_calculator):
        """Test basic calculator initialization with mock."""
        # Use the mock calculator from fixture
        assert hasattr(mock_calculator, 'elec_lr_cutoff')
        assert hasattr(mock_calculator, 'dtype')
        assert hasattr(mock_calculator, 'output_per_atom_energy_components')

    @pytest.mark.unit
    def test_calculator_initialization_failure(self, mock_calculator):
        """Test initialization with various parameter configurations."""
        # Test with invalid elec_lr_cutoff (negative value)
        mock_calculator.elec_lr_cutoff = -5.0
        assert mock_calculator.elec_lr_cutoff == -5.0

        # Test with extreme dtype
        mock_calculator.dtype = np.float16
        assert mock_calculator.dtype == np.float16

        # Test with components enabled/disabled toggling
        original_state = mock_calculator.output_per_atom_energy_components
        mock_calculator.output_per_atom_energy_components = not original_state
        assert mock_calculator.output_per_atom_energy_components == (not original_state)

        # Mock should still work regardless of parameter values
        energy = mock_calculator.calculate_energy("dummy_atoms")
        assert energy == -100.0

    @pytest.mark.unit
    def test_calculate_energy_basic(self, water_files, mock_calculator):
        """Test basic energy calculation with mock."""
        # Use the mock calculator directly from the fixture
        atoms = load_ase_structure(water_files['xyz'])[0]
        energy = mock_calculator.calculate_energy(atoms)

        # Mock calculator returns -100.0
        assert energy == -100.0
        assert isinstance(energy, (int, float))

    @pytest.mark.unit
    def test_get_per_atom_components_enabled(self, mock_calculator):
        """Test getting per-atom energy components with mock."""
        # Enable components and test
        mock_calculator.output_per_atom_energy_components = True
        components = mock_calculator.get_per_atom_energy_components()

        # Mock returns predefined components
        assert isinstance(components, dict)
        expected_keys = {'nn_energy', 'zbl_repulsion', 'electrostatic_energy', 'dispersion_energy'}
        assert set(components.keys()) == expected_keys

    @pytest.mark.unit
    def test_get_per_atom_components_disabled(self, mock_calculator):
        """Test per-atom components when disabled."""
        # Ensure components are disabled
        mock_calculator.output_per_atom_energy_components = False
        with pytest.raises(ValueError, match="Per-atom energy components not enabled"):
            mock_calculator.get_per_atom_energy_components()

    # ================================================================================================
    # INTEGRATION TESTS - REAL SO3LR CALCULATOR
    # ================================================================================================
    # The following tests use the actual SO3LR calculator with real model files.
    # These tests require:
    # 1. The so3lr package (with bundled model params) to be installed
    # 2. Real energy calculations with actual molecular structures
    # 3. Realistic energy ranges and validation
    #
    # These tests may be skipped if:
    # - The so3lr / JAX-MD stack is not installed
    # - Real calculations fail due to environment issues
    # - Dependencies are missing
    # ================================================================================================

    # @pytest.mark.integration
    @pytest.mark.slow
    def test_calculator_real_energy_water(self, water_files, real_calculator):
        """Test real energy calculation on water molecule."""
        try:

            atoms = load_ase_structure(water_files['xyz'])[0]

            energy = real_calculator.calculate_energy(atoms)

            # Water molecule energy should be in a reasonable range
            print(f"Calculated water energy: {energy}")
            assert isinstance(energy, (int, float))
            assert not np.isnan(energy)
            assert -500.0 < energy < 100.0  # Broad range for real energies

            print(f"Water energy (REAL): {energy:.4f}")

        except Exception as e:
            pytest.fail(f"SO3LR calculation failed: {e}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_calculator_real_energy_alanine(self, alanine_files, real_calculator):
        """Test real energy calculation on alanine molecule."""
        try:
            atoms = load_ase_structure(alanine_files['xyz'])[0]
            energy = real_calculator.calculate_energy(atoms)

            # Alanine energy should be more negative than water (larger molecule)
            assert isinstance(energy, (int, float))
            assert not np.isnan(energy)
            assert -1000.0 < energy < 100.0  # Broad range for alanine

            print(f"Alanine energy (REAL): {energy:.4f}")

        except Exception as e:
            pytest.fail(f"SO3LR calculation failed: {e}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_calculator_energy_comparison(self, water_files, alanine_files, real_calculator):
        """Test that larger molecules have more negative energies."""
        try:
            water_atoms = load_ase_structure(water_files['xyz'])[0]
            alanine_atoms = load_ase_structure(alanine_files['xyz'])[0]

            water_energy = real_calculator.calculate_energy(water_atoms)
            alanine_energy = real_calculator.calculate_energy(alanine_atoms)

            # Alanine (larger molecule) should have more negative energy than water
            assert alanine_energy < water_energy

            print(f"Water energy (REAL): {water_energy:.4f}")
            print(f"Alanine energy (REAL): {alanine_energy:.4f}")
            print(f"Energy difference: {alanine_energy - water_energy:.4f}")

        except Exception as e:
            pytest.fail(f"SO3LR calculation failed: {e}")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_calculator_per_atom_components_real(self, water_files):
        """Test per-atom energy components with real calculator."""
        try:
            calc = So3lrSfCalculator(
                output_per_atom_energy_components=True
            )

            atoms = load_ase_structure(water_files['xyz'])[0]
            energy = calc.calculate_energy(atoms)
            components = calc.get_per_atom_energy_components()

            # Should have components for each atom in water (3 atoms: O, H, H)
            assert isinstance(components, dict)
            assert len(components) > 0

            # Assert the specific component keys that SO3LR returns
            expected_keys = {'nn_energy', 'zbl_repulsion', 'electrostatic_energy', 'dispersion_energy'}
            actual_keys = set(components.keys())
            assert actual_keys == expected_keys, f"Expected {expected_keys}, but got {actual_keys}"

            # Each component should have same length as number of atoms
            for component_name, component_values in components.items():
                assert len(component_values) == len(atoms)
                assert isinstance(component_values, np.ndarray)

            print(f"Energy components for water (REAL): {list(components.keys())}")
            print(f"Total energy (REAL): {energy:.4f}")

        except Exception as e:
            pytest.fail(f"SO3LR calculation failed: {e}")

    @pytest.mark.unit
    def test_calculator_properties(self, mock_calculator):
        """Test calculator property access with mock."""
        # Set properties and test
        mock_calculator.elec_lr_cutoff = 15.0
        mock_calculator.dtype = np.float64
        mock_calculator.output_per_atom_energy_components = True

        assert mock_calculator.elec_lr_cutoff == 15.0
        assert mock_calculator.dtype == np.float64
        assert mock_calculator.output_per_atom_energy_components is True


class TestElectrostaticLrCutoff:
    """Tests for the electrostatic long-range cutoff (decoupled from the pinned dispersion cutoff)."""

    @pytest.mark.unit
    def test_stored_and_defaults(self):
        """elec_lr_cutoff defaults to 10 A; there is no lr_cutoff argument."""
        try:
            from src.calculator import So3lrSfCalculator, DISPERSION_LR_CUTOFF
        except ImportError as e:
            pytest.skip(f"so3lr / JAX-MD stack not available: {e}")

        assert So3lrSfCalculator().elec_lr_cutoff == 10.0
        assert So3lrSfCalculator(elec_lr_cutoff=8.0).elec_lr_cutoff == 8.0
        assert So3lrSfCalculator(elec_lr_cutoff=DISPERSION_LR_CUTOFF).elec_lr_cutoff == DISPERSION_LR_CUTOFF
        with pytest.raises(TypeError):
            So3lrSfCalculator(lr_cutoff=12.0)

        # The CLI default matches, declared inline on the argument.
        from so3lr_sf import setup_argument_parser
        args = setup_argument_parser().parse_args(["--protein", "p.pdb", "--ligands", "l.sdf"])
        assert args.elec_lr_cutoff == 10.0

    @pytest.mark.integration
    @pytest.mark.slow
    def test_equal_cutoff_reproduces_stock_so3lr_bitwise(self, water_files):
        """elec_lr_cutoff == DISPERSION_LR_CUTOFF must reproduce stock single-cutoff So3lrPotential."""
        try:
            import jax.numpy as jnp
            from jax_md import space
            from so3lr import So3lrPotential, to_jax_md
            from so3lr.jaxmd_utils import neighbor_list_featurizer
            from src.calculator import So3lrSfCalculator, DISPERSION_LR_CUTOFF
        except ImportError as e:
            pytest.skip(f"so3lr / JAX-MD stack not available: {e}")

        atoms = load_ase_structure(water_files['xyz'])[0]

        # Reference: stock So3lrPotential at the pinned dispersion cutoff.
        pos = jnp.asarray(atoms.get_positions(), dtype=jnp.float32)
        z = jnp.asarray(atoms.get_atomic_numbers(), dtype=jnp.int32)
        displacement, _ = space.free()
        potential = So3lrPotential(dtype=jnp.float32, lr_cutoff=DISPERSION_LR_CUTOFF)
        nfn, nfn_lr, _ = to_jax_md(
            potential=potential, displacement_or_metric=displacement, box_size=None,
            species=z, capacity_multiplier=1.25, buffer_size_multiplier_sr=1.25,
            buffer_size_multiplier_lr=1.25, minimum_cell_size_multiplier_sr=1.0,
            disable_cell_list=True, fractional_coordinates=False,
        )
        feat = neighbor_list_featurizer(displacement, z, fractional_coordinates=False)
        nbrs = nfn.allocate(pos, box=None)
        nbrs_lr = nfn_lr.allocate(pos, box=None)
        graph = feat(pos, nbrs.idx, nbrs_lr.idx, box=None)
        e_ref = float(potential(graph).sum())

        e_equal = So3lrSfCalculator(elec_lr_cutoff=DISPERSION_LR_CUTOFF).calculate_energy(atoms)
        assert e_equal == e_ref

    @pytest.mark.integration
    @pytest.mark.slow
    def test_shorter_cutoff_changes_energy(self, water_files):
        """A shorter electrostatic cutoff moves the energy vs the pinned 1000 A cutoff."""
        try:
            from src.calculator import So3lrSfCalculator, DISPERSION_LR_CUTOFF
        except ImportError as e:
            pytest.skip(f"so3lr / JAX-MD stack not available: {e}")

        atoms = load_ase_structure(water_files['xyz'])[0]

        e_wide = So3lrSfCalculator(elec_lr_cutoff=DISPERSION_LR_CUTOFF).calculate_energy(atoms)
        e_10 = So3lrSfCalculator(elec_lr_cutoff=10.0).calculate_energy(atoms)
        e_4 = So3lrSfCalculator(elec_lr_cutoff=4.0).calculate_energy(atoms)

        assert not np.isnan(e_10) and not np.isnan(e_4)
        assert abs(e_10 - e_wide) > 1e-3
        assert abs(e_4 - e_wide) > 1e-3
        print(f"water: elec@1000={e_wide:.6f}  elec@10={e_10:.6f}  elec@4={e_4:.6f}")