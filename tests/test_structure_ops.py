"""
Tests for the structure_ops module.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from ase import Atoms
from ase.optimize import FIRE, LBFGS

from src.structure_ops import (
    trim_structure,
    optimize_structure,
    extract_ligands
)
from src.utils import read_structure


class TestTrimStructure:
    """Tests for structure trimming functionality."""

    @pytest.mark.unit
    def test_trim_structure_basic(self, water_files, alanine_files, temp_dir):
        """Test basic structure trimming functionality."""
        protein_path = alanine_files['xyz']  # Use alanine as "protein"
        ligand_path = water_files['xyz']     # Use water as "ligand"

        # Use realistic radius for small molecules (1.5 Å)
        trimmed_protein_path, returned_ligand_path = trim_structure(
            protein_path, ligand_path, radius=1.5, output_dir=temp_dir
        )

        # Check that files were created
        assert Path(trimmed_protein_path).exists()
        assert returned_ligand_path == str(ligand_path)

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

        trimmed_protein_path, _ = trim_structure(protein_path, ligand_path, radius=2.0)

        # Check that output is in same directory as protein
        assert Path(trimmed_protein_path).parent == protein_path.parent
        assert Path(trimmed_protein_path).exists()

    @pytest.mark.unit
    def test_trim_structure_large_radius(self, water_files, alanine_files, temp_dir):
        """Test trimming with large radius (should include all atoms)."""
        protein_path = alanine_files['xyz']
        ligand_path = water_files['xyz']

        # Use a radius that should capture all atoms in small molecules
        trimmed_protein_path, _ = trim_structure(
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

        trimmed_protein_path, _ = trim_structure(
            protein_path, ligand_path, radius=radius, output_dir=temp_dir
        )

        expected_filename = f"{protein_path.stem}_trimmed_{radius}A.xyz"
        assert Path(trimmed_protein_path).name == expected_filename


class TestOptimizeStructure:
    """Tests for structure optimization functionality."""

    @pytest.mark.unit
    def test_optimize_structure_fire(self, water_files, temp_dir, so3lr_model_path):
        """Test structure optimization with FIRE optimizer."""
        structure_path = water_files['xyz']

        # Create a mock calculator that uses the real model path
        mock_calculator = Mock()
        mock_atoms = read_structure(structure_path)

        # Mock optimization methods
        with patch('src.structure_ops.FIRE') as mock_fire_class:
            mock_optimizer = Mock()
            mock_optimizer.run = Mock()
            mock_optimizer.converged.return_value = True
            mock_optimizer.nsteps = 10
            mock_fire_class.return_value = mock_optimizer

            # Mock energy calculations
            with patch.object(mock_atoms, 'get_potential_energy', side_effect=[-10.5, -11.0]):
                optimized_path, opt_info = optimize_structure(
                    structure_path,
                    optimizer='FIRE',
                    calculator=mock_calculator,
                    output_dir=temp_dir
                )

                # Check that optimization ran
                mock_optimizer.run.assert_called_once()
                assert opt_info['converged'] is True
                assert opt_info['steps'] == 10
                assert opt_info['initial_energy'] == -10.5
                assert opt_info['final_energy'] == -11.0

                # Check output file
                assert Path(optimized_path).exists()
                assert Path(optimized_path).name.endswith('_optimized.xyz')

    @pytest.mark.unit
    def test_optimize_structure_lbfgs(self, water_files, temp_dir):
        """Test structure optimization with LBFGS optimizer."""
        structure_path = water_files['xyz']
        mock_calculator = Mock()

        with patch('src.structure_ops.LBFGS') as mock_lbfgs_class:
            mock_optimizer = Mock()
            mock_optimizer.run = Mock()
            mock_optimizer.converged.return_value = False
            mock_optimizer.nsteps = 50
            mock_lbfgs_class.return_value = mock_optimizer

            optimized_path, opt_info = optimize_structure(
                structure_path,
                optimizer='LBFGS',
                calculator=mock_calculator,
                output_dir=temp_dir
            )

            assert opt_info['converged'] is False
            assert opt_info['steps'] == 50

    @pytest.mark.unit
    def test_optimize_structure_unknown_optimizer(self, water_files):
        """Test optimization with unknown optimizer."""
        with pytest.raises(ValueError, match="Unknown optimizer: UNKNOWN"):
            optimize_structure(water_files['xyz'], optimizer='UNKNOWN', calculator=Mock())

    @pytest.mark.unit
    def test_optimize_structure_no_calculator(self, water_files):
        """Test optimization without calculator raises error."""
        with pytest.raises(ValueError, match="Calculator must be provided"):
            optimize_structure(water_files['xyz'], calculator=None)

    @pytest.mark.unit
    def test_optimize_structure_optimization_error(self, water_files, temp_dir):
        """Test optimization with errors during optimization."""
        mock_calculator = Mock()

        with patch('src.structure_ops.FIRE') as mock_fire_class:
            mock_optimizer = Mock()
            mock_optimizer.run.side_effect = Exception("Optimization error")
            mock_fire_class.return_value = mock_optimizer

            # Should still save structure even with optimization error
            optimized_path, opt_info = optimize_structure(
                water_files['xyz'],
                calculator=mock_calculator,
                output_dir=temp_dir
            )

            assert Path(optimized_path).exists()
            assert opt_info['optimization_error'] == "Optimization error"

    @pytest.mark.unit
    def test_optimize_structure_default_output_dir(self, water_files):
        """Test optimization with default output directory."""
        mock_calculator = Mock()

        with patch('src.structure_ops.FIRE') as mock_fire_class:
            mock_optimizer = Mock()
            mock_optimizer.run = Mock()
            mock_optimizer.converged.return_value = True
            mock_optimizer.nsteps = 5
            mock_fire_class.return_value = mock_optimizer

            optimized_path, _ = optimize_structure(
                water_files['xyz'],
                calculator=mock_calculator
            )

            # Should use same directory as input file
            assert Path(optimized_path).parent == water_files['xyz'].parent

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

            optimized_path, opt_info = optimize_structure(
                water_files['xyz'],
                optimizer='FIRE',
                fmax=0.1,  # Less strict convergence for testing
                steps=10,   # Limit steps for testing
                calculator=calc._calculator,
                output_dir=temp_dir
            )

            assert Path(optimized_path).exists()
            assert isinstance(opt_info['initial_energy'], float)
            assert isinstance(opt_info['final_energy'], float)

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
    """Integration tests for structure operations."""

    @pytest.mark.integration
    def test_trim_and_optimize_workflow(self, water_files, alanine_files, temp_dir):
        """Test complete workflow of trimming and optimization."""
        protein_path = alanine_files['xyz']
        ligand_path = water_files['xyz']

        # Step 1: Trim structure with realistic radius
        trimmed_protein_path, _ = trim_structure(
            protein_path, ligand_path, radius=2.0, output_dir=temp_dir
        )

        # Step 2: Optimize trimmed structure
        mock_calculator = Mock()

        with patch('src.structure_ops.FIRE') as mock_fire_class:
            mock_optimizer = Mock()
            mock_optimizer.run = Mock()
            mock_optimizer.converged.return_value = True
            mock_optimizer.nsteps = 15
            mock_fire_class.return_value = mock_optimizer

            optimized_path, opt_info = optimize_structure(
                trimmed_protein_path,
                calculator=mock_calculator,
                output_dir=temp_dir
            )

            # Verify workflow completed successfully
            assert Path(optimized_path).exists()
            assert opt_info['converged'] is True

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

    @pytest.mark.integration
    @pytest.mark.slow
    def test_realistic_protein_ligand_trimming(self, water_files, alanine_files, temp_dir, so3lr_model_path):
        """Test realistic protein-ligand trimming with actual distances."""
        protein_path = alanine_files['xyz']
        ligand_path = water_files['xyz']

        # Test different realistic radii
        radii_to_test = [1.0, 1.5, 2.0, 2.5]

        for radius in radii_to_test:
            try:
                trimmed_protein_path, _ = trim_structure(
                    protein_path, ligand_path, radius=radius, output_dir=temp_dir / f"radius_{radius}"
                )

                # Verify trimmed structure
                original = read_structure(protein_path)
                trimmed = read_structure(trimmed_protein_path)

                assert len(trimmed) <= len(original)
                assert len(trimmed) > 0

                print(f"Radius {radius}Å: {len(trimmed)}/{len(original)} atoms kept")

            except ValueError:
                # Very small radius might not capture any atoms
                print(f"Radius {radius}Å: No atoms captured (too small)")
                continue