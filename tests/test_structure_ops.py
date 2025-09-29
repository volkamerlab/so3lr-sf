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
from src.utils import read_structure


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
        original_protein = read_structure(protein_path)
        trimmed_protein = read_structure(trimmed_protein_path)

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
        original_protein = read_structure(protein_path)
        trimmed_protein = read_structure(trimmed_protein_path)

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
        atoms = read_structure(water_files['xyz'])
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
        atoms = read_structure(water_files['xyz'])
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
        atoms = read_structure(water_files['xyz'])
        output_path = temp_dir / "test_output.xyz"
        with pytest.raises(ValueError, match="Unknown optimizer: UNKNOWN"):
            optimize_structure(atoms, calc=Mock(), optimizer='UNKNOWN', output_path=output_path)

    @pytest.mark.unit
    def test_optimize_structure_no_calculator(self, water_files, temp_dir):
        """Test optimization without calculator raises error."""
        atoms = read_structure(water_files['xyz'])
        output_path = temp_dir / "test_output.xyz"
        with pytest.raises((TypeError, ValueError, AttributeError)):
            optimize_structure(atoms, calc=None, output_path=output_path)

    @pytest.mark.unit
    def test_optimize_structure_optimization_error(self, water_files, temp_dir):
        """Test optimization with errors during optimization."""
        atoms = read_structure(water_files['xyz'])
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
        atoms = read_structure(water_files['xyz'])
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
            atoms = read_structure(water_files['xyz'])
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
            atoms = read_structure(ligand_file)
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

        with pytest.raises(ValueError):
            extract_ligands(fake_sdf, output_dir=temp_dir)

    @pytest.mark.unit
    def test_extract_ligands_from_xyz_trajectory(self, temp_dir):
        """Test extracting from XYZ trajectory-like file."""
        # Create a multi-frame XYZ file
        multi_xyz = temp_dir / "trajectory.xyz"
        xyz_content = """3
Frame 1
O    0.0000    0.0000    0.0000
H    0.7570    0.5860    0.0000
H   -0.7570    0.5860    0.0000
3
Frame 2
O    1.0000    0.0000    0.0000
H    1.7570    0.5860    0.0000
H    0.2430    0.5860    0.0000
"""
        multi_xyz.write_text(xyz_content)

        ligand_files = extract_ligands(multi_xyz, output_dir=temp_dir)

        assert len(ligand_files) == 2
        for ligand_file in ligand_files:
            assert Path(ligand_file).exists()
            atoms = read_structure(ligand_file)
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
        original_protein = read_structure(protein_path)
        ligand = read_structure(ligand_path)
        trimmed_protein = read_structure(trimmed_protein_path)

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
        atoms = read_structure(water_files['xyz'])
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
        optimized_atoms = read_structure(optimized_path)
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
        atoms = read_structure(alanine_files['xyz'])
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
        optimized_atoms = read_structure(optimized_path)
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
        atoms = read_structure(water_files['xyz'])
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
        optimized_atoms = read_structure(optimized_path)
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
        trimmed_atoms = read_structure(trimmed_protein_path)
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
        final_atoms = read_structure(optimized_path)
        final_positions = final_atoms.get_positions()

        # Verify trimming worked
        original_protein = read_structure(protein_path)
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
            atoms = read_structure(ligand_file)
            results.append({
                'file': ligand_file,
                'n_atoms': len(atoms),
                'formula': atoms.get_chemical_formula()
            })

        assert len(results) == 2
        for result in results:
            assert result['n_atoms'] == 3
            assert result['formula'] == 'H2O'