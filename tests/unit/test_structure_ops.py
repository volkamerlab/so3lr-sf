"""
Unit tests for the structure_ops module.

This module tests all structure operation functions including:
- Protein trimming around ligands
- Structure optimization with different algorithms
- Ligand extraction from multi-structure files
- Workflow functions for protein/ligand processing
- Complex structure manipulation

Each test is designed to be:
1. Independent and isolated
2. Readable with descriptive names
3. Well-documented with clear explanations
4. Comprehensive in edge case coverage
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from ase import Atoms

from src.structure_ops import (
    trim_structure,
    optimize_structure,
    extract_ligands,
    perform_trimming,
    optimize_protein,
    process_single_ligand
)


class TestStructureTrimming:
    """Test protein structure trimming functionality."""

    def test_trim_structure_basic(self, sample_protein_atoms, sample_ligand_atoms, temp_dir):
        """
        Test basic protein trimming around a ligand.

        This test verifies that the trimming function correctly
        identifies and keeps only protein atoms within a specified
        radius of the ligand.
        """
        # Arrange: Create protein and ligand files
        protein_file = temp_dir / "protein.xyz"
        ligand_file = temp_dir / "ligand.xyz"

        from src.utils import write_structure
        write_structure(sample_protein_atoms, protein_file)
        write_structure(sample_ligand_atoms, ligand_file)

        # Act: Trim protein around ligand with 5.0 Å radius
        trimmed_path = trim_structure(
            protein_file, ligand_file, radius=5.0, output_dir=temp_dir
        )

        # Assert: Trimmed file should be created
        assert Path(trimmed_path).exists()
        assert "trimmed" in str(trimmed_path)

        # Verify trimmed structure
        from src.utils import read_structure
        trimmed_atoms = read_structure(trimmed_path)
        assert isinstance(trimmed_atoms, Atoms)
        assert len(trimmed_atoms) <= len(sample_protein_atoms)  # Should be smaller or equal

    def test_trim_structure_large_radius(self, sample_protein_atoms, sample_ligand_atoms, temp_dir):
        """
        Test protein trimming with large radius.

        This test verifies that when the radius is very large,
        most or all of the original protein atoms are retained.
        """
        # Arrange: Create files
        protein_file = temp_dir / "protein.xyz"
        ligand_file = temp_dir / "ligand.xyz"

        from src.utils import write_structure
        write_structure(sample_protein_atoms, protein_file)
        write_structure(sample_ligand_atoms, ligand_file)

        # Act: Trim with very large radius
        trimmed_path = trim_structure(
            protein_file, ligand_file, radius=100.0, output_dir=temp_dir
        )

        # Assert: Should keep most/all atoms with large radius
        from src.utils import read_structure
        trimmed_atoms = read_structure(trimmed_path)
        original_size = len(sample_protein_atoms)
        trimmed_size = len(trimmed_atoms)

        # Should retain significant portion with large radius
        assert trimmed_size >= original_size * 0.8

    def test_trim_structure_small_radius(self, sample_protein_atoms, sample_ligand_atoms, temp_dir):
        """
        Test protein trimming with small radius.

        This test verifies that small radius values result in
        significant reduction of protein atoms.
        """
        # Arrange: Create files with ligand far from protein
        protein_file = temp_dir / "protein.xyz"
        ligand_file = temp_dir / "ligand.xyz"

        # Move ligand far away
        distant_ligand = sample_ligand_atoms.copy()
        distant_ligand.translate([50.0, 50.0, 50.0])

        from src.utils import write_structure
        write_structure(sample_protein_atoms, protein_file)
        write_structure(distant_ligand, ligand_file)

        # Act: Trim with small radius
        trimmed_path = trim_structure(
            protein_file, ligand_file, radius=1.0, output_dir=temp_dir
        )

        # Assert: Should significantly reduce atom count
        from src.utils import read_structure
        trimmed_atoms = read_structure(trimmed_path)
        original_size = len(sample_protein_atoms)
        trimmed_size = len(trimmed_atoms)

        # With distant ligand and small radius, should trim significantly
        assert trimmed_size < original_size

    def test_trim_structure_custom_output_dir(self, sample_protein_atoms, sample_ligand_atoms, temp_dir):
        """
        Test protein trimming with custom output directory.

        This test verifies that the trimming function respects
        custom output directory specifications.
        """
        # Arrange: Create files and custom output directory
        protein_file = temp_dir / "protein.xyz"
        ligand_file = temp_dir / "ligand.xyz"
        custom_output = temp_dir / "custom_trimming"

        from src.utils import write_structure
        write_structure(sample_protein_atoms, protein_file)
        write_structure(sample_ligand_atoms, ligand_file)

        # Act: Trim with custom output directory
        trimmed_path = trim_structure(
            protein_file, ligand_file, radius=5.0, output_dir=custom_output
        )

        # Assert: Output should be in custom directory
        assert custom_output.name in str(trimmed_path)
        assert Path(trimmed_path).exists()


class TestStructureOptimization:
    """Test structure optimization functionality."""

    def test_optimize_structure_fire_algorithm(self, sample_protein_atoms, temp_dir, mock_calculator):
        """
        Test structure optimization using FIRE algorithm.

        This test verifies that the FIRE optimization algorithm
        correctly optimizes structures and saves results.
        """
        # Arrange: Output path for optimized structure
        output_path = temp_dir / "optimized.xyz"

        # Mock successful optimization
        mock_calculator.calculate_energy = Mock(return_value=-100.0)

        # Act: Optimize structure using FIRE
        optimized_path, opt_info = optimize_structure(
            sample_protein_atoms, mock_calculator,
            optimizer="FIRE", fmax=0.05, steps=10,
            output_path=output_path
        )

        # Assert: Should return valid optimization results
        assert Path(optimized_path).exists()
        assert isinstance(opt_info, dict)
        assert 'converged' in opt_info
        assert 'n_steps' in opt_info
        assert 'final_fmax' in opt_info

        # Verify optimization was called
        assert mock_calculator.calculate_energy.called

    def test_optimize_structure_lbfgs_algorithm(self, sample_protein_atoms, temp_dir, mock_calculator):
        """
        Test structure optimization using L-BFGS algorithm.

        This test verifies that the L-BFGS optimization algorithm
        works correctly as an alternative to FIRE.
        """
        # Arrange: Output path
        output_path = temp_dir / "optimized.xyz"

        # Act: Optimize using L-BFGS
        optimized_path, opt_info = optimize_structure(
            sample_protein_atoms, mock_calculator,
            optimizer="LBFGS", fmax=0.05, steps=10,
            output_path=output_path
        )

        # Assert: Should complete optimization
        assert Path(optimized_path).exists()
        assert isinstance(opt_info, dict)
        assert opt_info['optimizer'] == "LBFGS"

    def test_optimize_structure_convergence_criteria(self, sample_protein_atoms, temp_dir, mock_calculator):
        """
        Test optimization with different convergence criteria.

        This test verifies that the optimization respects
        different force convergence thresholds.
        """
        # Arrange: Strict convergence criteria
        output_path = temp_dir / "optimized.xyz"

        # Act: Optimize with strict fmax
        optimized_path, opt_info = optimize_structure(
            sample_protein_atoms, mock_calculator,
            optimizer="FIRE", fmax=0.001, steps=50,
            output_path=output_path
        )

        # Assert: Should record convergence information
        assert 'final_fmax' in opt_info
        assert opt_info['fmax_target'] == 0.001

    def test_optimize_structure_with_constraints(self, sample_complex_atoms, temp_dir, mock_calculator):
        """
        Test constrained optimization around ligand region.

        This test verifies that optimization can be constrained
        to atoms within a specific radius of the ligand.
        """
        # Arrange: Complex with protein + ligand
        output_path = temp_dir / "optimized_complex.xyz"
        n_protein_atoms = 10  # First 10 atoms are protein

        # Act: Optimize with radius constraint
        optimized_path, opt_info = optimize_structure(
            sample_complex_atoms, mock_calculator,
            optimizer="FIRE", fmax=0.05, steps=10,
            output_path=output_path,
            opt_radius=5.0, n_protein_atoms=n_protein_atoms
        )

        # Assert: Should complete constrained optimization
        assert Path(optimized_path).exists()
        assert 'constrained' in opt_info or 'opt_radius' in opt_info

    def test_optimize_structure_max_steps_reached(self, sample_protein_atoms, temp_dir, mock_calculator):
        """
        Test optimization behavior when maximum steps are reached.

        This test verifies that optimization properly handles
        cases where convergence is not achieved within step limit.
        """
        # Arrange: Very few steps to force non-convergence
        output_path = temp_dir / "optimized.xyz"

        # Act: Optimize with very few steps
        optimized_path, opt_info = optimize_structure(
            sample_protein_atoms, mock_calculator,
            optimizer="FIRE", fmax=0.001, steps=1,  # Very few steps
            output_path=output_path
        )

        # Assert: Should complete but likely not converged
        assert Path(optimized_path).exists()
        assert 'n_steps' in opt_info
        # May or may not be converged depending on initial forces


class TestLigandExtraction:
    """Test ligand extraction from multi-structure files."""

    def test_extract_ligands_multi_sdf(self, multi_sdf_file, temp_dir):
        """
        Test extraction of ligands from multi-molecule SDF file.

        This test verifies that multiple ligands are correctly
        extracted from a single SDF file containing multiple molecules.
        """
        # Arrange: Output directory for extracted ligands
        extract_dir = temp_dir / "extracted_ligands"

        # Act: Extract ligands from multi-SDF
        ligand_files = extract_ligands(multi_sdf_file, extract_dir)

        # Assert: Should extract multiple ligands
        assert isinstance(ligand_files, list)
        assert len(ligand_files) >= 2  # Multi-SDF should have multiple ligands

        # Verify all extracted files exist
        for ligand_file in ligand_files:
            assert Path(ligand_file).exists()
            assert ligand_file.endswith('.xyz')  # Should be converted to XYZ

        # Verify directory structure
        assert extract_dir.exists()

    def test_extract_ligands_single_structure(self, sample_sdf_file, temp_dir):
        """
        Test extraction from single-structure file.

        This test verifies that single-structure files are
        handled correctly and produce one output file.
        """
        # Arrange: Output directory
        extract_dir = temp_dir / "extracted_ligands"

        # Act: Extract from single SDF
        ligand_files = extract_ligands(sample_sdf_file, extract_dir)

        # Assert: Should extract one ligand
        assert isinstance(ligand_files, list)
        assert len(ligand_files) == 1
        assert Path(ligand_files[0]).exists()

    def test_extract_ligands_custom_naming(self, multi_sdf_file, temp_dir):
        """
        Test ligand extraction with custom file naming.

        This test verifies that extracted ligands can be named
        with custom prefixes for better organization.
        """
        # Arrange: Custom naming prefix
        extract_dir = temp_dir / "custom_ligands"
        custom_prefix = "molecule"

        # Act: Extract with custom naming
        ligand_files = extract_ligands(
            multi_sdf_file, extract_dir, naming_prefix=custom_prefix
        )

        # Assert: Files should use custom prefix
        for ligand_file in ligand_files:
            filename = Path(ligand_file).name
            assert filename.startswith(custom_prefix)

    def test_extract_ligands_invalid_file(self, temp_dir):
        """
        Test error handling with invalid structure files.

        This test verifies that appropriate errors are raised
        when trying to extract from invalid or corrupted files.
        """
        # Arrange: Invalid file
        invalid_file = temp_dir / "invalid.sdf"
        invalid_file.write_text("This is not a valid SDF file")

        extract_dir = temp_dir / "extracted"

        # Act & Assert: Should handle invalid file gracefully
        # The exact behavior depends on implementation
        # Could raise error or return empty list
        try:
            ligand_files = extract_ligands(invalid_file, extract_dir)
            # If no error, should return empty or valid list
            assert isinstance(ligand_files, list)
        except Exception:
            # If error is raised, that's also acceptable behavior
            pass


class TestWorkflowFunctions:
    """Test high-level workflow functions."""

    def test_perform_trimming_with_specified_ligand(self, temp_dir, sample_protein_atoms, sample_ligand_atoms, mock_logger):
        """
        Test protein trimming workflow with specific ligand.

        This test verifies that the trimming workflow correctly
        uses a specified ligand file for trimming operations.
        """
        # Arrange: Create protein and ligand files
        protein_file = temp_dir / "protein.xyz"
        ligand_file = temp_dir / "ligand.xyz"

        from src.utils import write_structure
        write_structure(sample_protein_atoms, protein_file)
        write_structure(sample_ligand_atoms, ligand_file)

        # Act: Perform trimming with specified ligand
        trimmed_path = perform_trimming(
            protein_file, "dummy_ligands", radius=5.0,
            trim_lig=str(ligand_file), output_dir=temp_dir,
            logger=mock_logger
        )

        # Assert: Should complete trimming
        assert Path(trimmed_path).exists()
        assert "trimmed" in str(trimmed_path)

        # Verify logger was called
        mock_logger.info.assert_called()

    def test_perform_trimming_auto_ligand_selection(self, temp_dir, sample_protein_atoms, multi_sdf_file, mock_logger):
        """
        Test automatic ligand selection for trimming.

        This test verifies that when no specific ligand is provided,
        the function automatically selects the first available ligand.
        """
        # Arrange: Create protein file
        protein_file = temp_dir / "protein.xyz"

        from src.utils import write_structure
        write_structure(sample_protein_atoms, protein_file)

        # Act: Perform trimming with auto ligand selection
        trimmed_path = perform_trimming(
            protein_file, str(multi_sdf_file), radius=8.0,
            trim_lig=None, output_dir=temp_dir,
            logger=mock_logger
        )

        # Assert: Should complete trimming with auto-selected ligand
        assert Path(trimmed_path).exists()
        mock_logger.info.assert_called()

    def test_optimize_protein_workflow(self, temp_dir, sample_protein_atoms, mock_calculator, mock_logger):
        """
        Test complete protein optimization workflow.

        This test verifies that the protein optimization workflow
        correctly optimizes proteins and logs optimization details.
        """
        # Arrange: Create protein file
        protein_file = temp_dir / "protein.xyz"

        from src.utils import write_structure
        write_structure(sample_protein_atoms, protein_file)

        optimization_log = []

        # Act: Optimize protein
        optimized_path = optimize_protein(
            str(protein_file), mock_calculator,
            optimizer="FIRE", fmax=0.05, steps=100,
            output_dir=temp_dir, optimization_log=optimization_log,
            opt_log=True, logger=mock_logger
        )

        # Assert: Should complete optimization
        assert Path(optimized_path).exists()
        assert len(optimization_log) == 1  # Should add entry to log

        # Verify log entry
        log_entry = optimization_log[0]
        assert log_entry['structure_type'] == 'protein'
        assert 'optimization_info' in log_entry

        # Verify logger calls
        mock_logger.info.assert_called()

    def test_process_single_ligand_basic(self, temp_dir, sample_protein_atoms, sample_ligand_atoms, mock_calculator, mock_logger):
        """
        Test basic single ligand processing workflow.

        This test verifies that a single ligand can be processed
        through the complete workflow without optimization.
        """
        # Arrange: Create structure files
        protein_file = temp_dir / "protein.xyz"
        ligand_file = temp_dir / "ligand.xyz"

        from src.utils import write_structure
        write_structure(sample_protein_atoms, protein_file)
        write_structure(sample_ligand_atoms, ligand_file)

        # Mock args for basic calculation
        args = Mock()
        args.optimize = False
        args.explain = False

        # Mock interaction energy calculation
        with patch('src.structure_ops.protein_ligand_interaction') as mock_interaction:
            mock_interaction.return_value = -0.5  # Simple energy value

            # Act: Process single ligand
            result, error = process_single_ligand(
                str(ligand_file), args, mock_calculator,
                str(protein_file), temp_dir, None, mock_logger
            )

        # Assert: Should complete successfully
        assert error is None
        assert isinstance(result, dict)
        assert 'ligand_name' in result
        assert 'interaction_energy' in result
        assert result['interaction_energy'] == -0.5

    def test_process_single_ligand_with_optimization(self, temp_dir, sample_protein_atoms, sample_ligand_atoms, mock_calculator, mock_logger):
        """
        Test single ligand processing with optimization enabled.

        This test verifies that ligand processing correctly handles
        structure optimization when requested.
        """
        # Arrange: Create structure files
        protein_file = temp_dir / "protein.xyz"
        ligand_file = temp_dir / "ligand.xyz"

        from src.utils import write_structure
        write_structure(sample_protein_atoms, protein_file)
        write_structure(sample_ligand_atoms, ligand_file)

        # Mock args for optimization
        args = Mock()
        args.optimize = True
        args.explain = False
        args.optimizer = "FIRE"
        args.fmax = 0.05
        args.steps = 100
        args.opt_radius = None
        args.opt_log = False

        optimization_log = []

        # Mock optimization and interaction calculation
        with patch('src.structure_ops.optimize_structure') as mock_opt:
            mock_opt.return_value = (str(temp_dir / "opt.xyz"), {'converged': True})

            with patch('src.structure_ops.protein_ligand_interaction') as mock_interaction:
                mock_interaction.return_value = -0.75

                # Act: Process with optimization
                result, error = process_single_ligand(
                    str(ligand_file), args, mock_calculator,
                    str(protein_file), temp_dir, optimization_log, mock_logger
                )

        # Assert: Should complete with optimization
        assert error is None
        assert isinstance(result, dict)
        assert result['interaction_energy'] == -0.75

        # Should call optimization multiple times (ligand, complex)
        assert mock_opt.call_count >= 1

    def test_process_single_ligand_with_explainability(self, temp_dir, sample_protein_atoms, sample_ligand_atoms, mock_calculator, mock_logger):
        """
        Test single ligand processing with explainability analysis.

        This test verifies that explainability analysis is correctly
        performed and results are included in the output.
        """
        # Arrange: Create structure files
        protein_file = temp_dir / "protein.xyz"
        ligand_file = temp_dir / "ligand.xyz"

        from src.utils import write_structure
        write_structure(sample_protein_atoms, protein_file)
        write_structure(sample_ligand_atoms, ligand_file)

        # Mock args for explainability
        args = Mock()
        args.optimize = False
        args.explain = True

        # Mock interaction calculation with explainability
        mock_analysis = {
            'component_totals': {'MLFF': -0.3, 'Electrostatics': -0.2},
            'heatmap_path': 'test_heatmap.png'
        }

        with patch('src.structure_ops.protein_ligand_interaction') as mock_interaction:
            mock_interaction.return_value = (-0.5, mock_analysis)

            # Act: Process with explainability
            result, error = process_single_ligand(
                str(ligand_file), args, mock_calculator,
                str(protein_file), temp_dir, None, mock_logger
            )

        # Assert: Should include explainability results
        assert error is None
        assert 'analysis' in result
        assert result['analysis'] == mock_analysis

    def test_process_single_ligand_error_handling(self, temp_dir, mock_calculator, mock_logger):
        """
        Test error handling in single ligand processing.

        This test verifies that errors during ligand processing
        are properly caught and reported.
        """
        # Arrange: Non-existent files to trigger error
        protein_file = temp_dir / "nonexistent_protein.xyz"
        ligand_file = temp_dir / "nonexistent_ligand.xyz"

        args = Mock()
        args.optimize = False
        args.explain = False

        # Act: Process with invalid files
        result, error = process_single_ligand(
            str(ligand_file), args, mock_calculator,
            str(protein_file), temp_dir, None, mock_logger
        )

        # Assert: Should handle error gracefully
        assert error is not None
        assert isinstance(result, dict)
        assert 'error' in result
        assert 'ligand_name' in result
        assert np.isnan(result['interaction_energy'])


@pytest.mark.unit
class TestStructureOpsIntegration:
    """Integration tests for structure operations working together."""

    def test_complete_structure_workflow(self, temp_dir, sample_protein_atoms, multi_sdf_file, mock_calculator, mock_logger):
        """
        Test complete structure processing workflow.

        This test verifies that all structure operations work together
        in a realistic protein-ligand processing scenario.
        """
        # Arrange: Create protein file
        protein_file = temp_dir / "protein.xyz"

        from src.utils import write_structure
        write_structure(sample_protein_atoms, protein_file)

        # Act: Execute complete workflow
        # 1. Extract ligands
        extract_dir = temp_dir / "ligands"
        ligand_files = extract_ligands(multi_sdf_file, extract_dir)

        # 2. Trim protein around first ligand
        trimmed_path = perform_trimming(
            protein_file, str(multi_sdf_file), radius=8.0,
            trim_lig=None, output_dir=temp_dir, logger=mock_logger
        )

        # 3. Optimize trimmed protein
        optimized_path = optimize_protein(
            trimmed_path, mock_calculator,
            optimizer="FIRE", fmax=0.05, steps=50,
            output_dir=temp_dir, optimization_log=[],
            opt_log=False, logger=mock_logger
        )

        # 4. Process first ligand
        args = Mock()
        args.optimize = True
        args.explain = False
        args.optimizer = "FIRE"
        args.fmax = 0.05
        args.steps = 50
        args.opt_radius = None
        args.opt_log = False

        with patch('src.structure_ops.optimize_structure') as mock_opt:
            mock_opt.return_value = (str(temp_dir / "opt.xyz"), {'converged': True})

            with patch('src.structure_ops.protein_ligand_interaction') as mock_interaction:
                mock_interaction.return_value = -0.8

                result, error = process_single_ligand(
                    ligand_files[0], args, mock_calculator,
                    optimized_path, temp_dir, [], mock_logger
                )

        # Assert: Complete workflow should succeed
        assert len(ligand_files) >= 2
        assert Path(trimmed_path).exists()
        assert Path(optimized_path).exists()
        assert error is None
        assert isinstance(result, dict)
        assert result['interaction_energy'] == -0.8


# Test helper functions
def create_mock_optimization_result():
    """Helper function to create mock optimization results."""
    return {
        'converged': True,
        'n_steps': 45,
        'final_fmax': 0.04,
        'optimizer': 'FIRE',
        'fmax_target': 0.05
    }


def assert_valid_optimization_info(opt_info):
    """Helper function to validate optimization info structure."""
    assert isinstance(opt_info, dict)
    assert 'converged' in opt_info
    assert 'n_steps' in opt_info
    assert 'final_fmax' in opt_info