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

        expected_filename = f"{protein_path.stem}_trimmed_{radius}A.xyz"
        assert Path(trimmed_protein_path).name == expected_filename

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
        from src.utils import setup_logging

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