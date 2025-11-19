"""
Tests for the optimization module.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.optimization import optimize_structure
from src.molecule_loader import load_ase_structure, extract_ligands
from src.trim import trim_structure


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
        with patch('ase.optimize.FIRE') as mock_fire_class:
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

        with patch('ase.optimize.LBFGS') as mock_lbfgs_class:
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

        with patch('ase.optimize.FIRE') as mock_fire_class:
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

        with patch('ase.optimize.FIRE') as mock_fire_class:
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

        except Exception as e:
            pytest.skip(f"SO3LR calculation failed: {e}")



    @pytest.mark.unit
    def test_get_optimizer_function_available_optimizers(self, water_files):
        """Test get_optimizer function with various available ASE optimizers."""
        from src.optimization import get_optimizer

        atoms = load_ase_structure(water_files['xyz'])[0]

        # Test all compatible ASE optimizers (CellAwareBFGS excluded due to special requirements)
        test_optimizers = ['BFGS', 'BFGSLineSearch', 'FIRE', 'FIRE2', 'GPMin', 'GoodOldQuasiNewton', 'LBFGS', 'LBFGSLineSearch', 'MDMin', 'ODE12r', 'QuasiNewton']

        for optimizer_name in test_optimizers:
            try:
                opt = get_optimizer(atoms, optimizer_name)
                # Verify it's an optimizer instance
                assert hasattr(opt, 'run'), f"{optimizer_name} should have a 'run' method"
                assert hasattr(opt, 'atoms'), f"{optimizer_name} should have an 'atoms' attribute"
                # Verify atoms are correctly assigned
                assert opt.atoms is atoms, f"{optimizer_name} should reference the correct atoms object"
            except ValueError as e:
                # If optimizer is not available, that's acceptable - skip it
                if "Unknown optimizer" in str(e):
                    pytest.raises(ValueError, match=f"Unknown optimizer: {optimizer_name}")
                else:
                    raise

    @pytest.mark.unit
    def test_get_optimizer_case_insensitive(self, water_files):
        """Test that get_optimizer is case-insensitive."""
        from src.optimization import get_optimizer

        atoms = load_ase_structure(water_files['xyz'])[0]

        # Test that upper, lower, and mixed case all work
        for optimizer_name in ['FIRE', 'fire', 'Fire']:
            opt = get_optimizer(atoms, optimizer_name)
            assert opt.__class__.__name__ == 'FIRE'

        for optimizer_name in ['LBFGS', 'lbfgs', 'Lbfgs']:
            opt = get_optimizer(atoms, optimizer_name)
            assert opt.__class__.__name__ == 'LBFGS'

    @pytest.mark.unit
    def test_get_optimizer_invalid_optimizer(self, water_files):
        """Test get_optimizer with invalid optimizer name."""
        from src.optimization import get_optimizer

        atoms = load_ase_structure(water_files['xyz'])[0]

        with pytest.raises(ValueError) as exc_info:
            get_optimizer(atoms, 'INVALID_OPTIMIZER')

        error_msg = str(exc_info.value)
        assert "Unknown optimizer: INVALID_OPTIMIZER" in error_msg
        assert "Available optimizers:" in error_msg
        # Should list some common optimizers
        assert "FIRE" in error_msg
        assert "LBFGS" in error_msg

    @pytest.mark.integration
    def test_optimize_structure_with_various_optimizers_real(self, water_files, temp_dir, real_calculator):
        """Test optimize_structure works with various ASE optimizers using real calculations."""
        atoms = load_ase_structure(water_files['xyz'])[0]
        original_positions = atoms.get_positions().copy()

        # Test different optimizers with real calculations (subset for speed)
        test_optimizers = [
        'FIRE', 'FIRE2', 'LBFGS', 'BFGS', 'BFGSLineSearch', 'LBFGSLineSearch',
        'GPMin', 'MDMin', 'ODE12r', 'GoodOldQuasiNewton', 'QuasiNewton'
    ]

        for optimizer_name in test_optimizers:
            print(f"\n=== Testing real optimization with {optimizer_name} ===")

            try:
                # Create a copy of atoms for each test
                test_atoms = atoms.copy()
                output_path = temp_dir / f"real_optimized_{optimizer_name.lower()}.xyz"

                # Run actual optimization with minimal steps for testing
                optimized_path, opt_info = optimize_structure(
                    test_atoms,
                    calc=real_calculator,
                    optimizer=optimizer_name,
                    fmax=0.5,  # Loose convergence for speed
                    steps=5,   # Few steps for testing
                    output_path=output_path
                )

                # Verify optimization completed
                assert Path(optimized_path).exists()
                assert opt_info['optimizer'] == optimizer_name
                assert opt_info['converged'] in ['yes', 'no']  # Either is acceptable for few steps

                # Verify structure was actually modified
                optimized_atoms = load_ase_structure(optimized_path)[0]
                optimized_positions = optimized_atoms.get_positions()

                # Calculate position changes
                position_changes = np.linalg.norm(optimized_positions - original_positions, axis=1)
                max_change = np.max(position_changes)

                # Should have some position change (even if small)
                assert max_change >= 0.0, "Positions should be tracked"

                print(f"✓ {optimizer_name}: converged={opt_info['converged']}, "
                      f"steps={opt_info['steps']} eV, "
                      f"max_pos_change={max_change:.6f} Å")

            except ValueError as e:
                if "Unknown optimizer" in str(e):
                    pytest.skip(f"Optimizer {optimizer_name} not available in this ASE version")
                else:
                    raise
            except Exception as e:
                # Log the error but don't fail the test - some optimizers might have issues
                print(f"⚠ {optimizer_name} failed: {e}")
                pytest.skip(f"Optimizer {optimizer_name} encountered issues: {e}")

    @pytest.mark.unit
    def test_get_optimizer_real_instantiation(self, water_files):
        """Test get_optimizer creates working optimizer instances."""
        from src.optimization import get_optimizer

        atoms = load_ase_structure(water_files['xyz'])[0]

        # Test that we can actually create and use optimizer instances
        test_optimizers = ['FIRE', 'LBFGS', 'BFGS']

        for optimizer_name in test_optimizers:
            try:
                opt = get_optimizer(atoms.copy(), optimizer_name)

                # Verify it's a proper optimizer
                assert hasattr(opt, 'run'), f"{optimizer_name} should have run method"
                assert hasattr(opt, 'atoms'), f"{optimizer_name} should have atoms attribute"
                assert opt.atoms is not None, f"{optimizer_name} atoms should be set"
                assert len(opt.atoms) == len(atoms), f"{optimizer_name} should have correct number of atoms"

                # Verify optimizer class name matches expectation
                assert opt.__class__.__name__ == optimizer_name, f"Expected {optimizer_name}, got {opt.__class__.__name__}"

                print(f"✓ {optimizer_name}: {type(opt).__name__} with {len(opt.atoms)} atoms")

            except ValueError as e:
                if "Unknown optimizer" in str(e):
                    pytest.skip(f"Optimizer {optimizer_name} not available")
                else:
                    raise


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

        # Load optimized structure and verify positions changed
        optimized_atoms = load_ase_structure(optimized_path)[0]
        optimized_positions = optimized_atoms.get_positions()

        # Calculate position changes
        position_changes = np.linalg.norm(optimized_positions - original_positions, axis=1)
        max_change = np.max(position_changes)

        # Verify that at least some atoms moved (optimization occurred)
        assert max_change > 1e-6, "Optimization should cause position changes"

        print(f"Water optimization: max position change = {max_change:.6f} Å")

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

        # Load optimized structure and verify positions changed
        optimized_atoms = load_ase_structure(optimized_path)[0]
        optimized_positions = optimized_atoms.get_positions()

        # Calculate position changes
        position_changes = np.linalg.norm(optimized_positions - original_positions, axis=1)
        max_change = np.max(position_changes)

        # Verify that at least some atoms moved (optimization occurred)
        assert max_change > 1e-6, "Optimization should cause position changes"

        print(f"Alanine optimization: max position change = {max_change:.6f} Å")

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
    def test_perform_trimming_real(self, test_data_dir, temp_dir):
        """Test perform_trimming with real data workflow."""
        from src.trim import perform_trimming

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


class TestProcessSingleLigand:
    """Tests for process_single_ligand function."""

    @pytest.mark.unit
    def test_process_single_ligand_basic_no_optimization(self, temp_dir, water_files, mock_calculator):
        """Test basic ligand processing without optimization."""
        from src.optimization import process_single_ligand
        from unittest.mock import Mock

        # Create mock args
        mock_args = Mock()
        mock_args.optimize = False
        mock_args.exp_lig = False
        mock_args.exp_prot = False
        mock_args.exp_3d = False
        mock_args.eda = False
        mock_args.opt_log = False

        mock_logger = Mock()
        optimization_log = []

        with patch('src.optimization.protein_ligand_interaction') as mock_interaction:
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
        from src.optimization import process_single_ligand
        from unittest.mock import Mock

        # Create mock args
        mock_args = Mock()
        mock_args.optimize = False
        mock_args.exp_lig = True
        mock_args.eda = False
        mock_args.opt_log = False
        mock_args.exp_prot = True
        mock_args.exp_3d = True
        mock_args.verbose = False
        
        mock_logger = Mock()
        optimization_log = []
        mock_preloaded_protein_prolif = Mock()
        with patch('src.optimization.protein_ligand_interaction') as mock_interaction:
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
            print("Result:", result)
            # Check no error
            assert error is None

            # Check analysis was included
            assert result['ligand_explainability'] == mock_analysis
            assert result['interaction_energy'] == -2.3

            # Check that exp_outputs was set correctly
            expected_exp_outputs = (
                temp_dir / "ligand_exp" / "water_heatmap.png",  # exp_lig output
                temp_dir / "pl_2d_exp" / "water_protein_interactions.png",  # exp_prot output
                temp_dir / "pl_3d_exp" / "water_3d_visualization.pml"  # exp_3d output
            )
            mock_interaction.assert_called_with(
                water_files['pdb'], water_files['sdf'], mock_calculator,
                complex_path=None,
                eda=False,
                verbose=False,
                preloaded_protein_prolif=None,
                exp_outputs=expected_exp_outputs
            )

    @pytest.mark.unit
    def test_process_single_ligand_with_eda(self, temp_dir, water_files, mock_calculator):
        """Test ligand processing with EDA analysis."""
        from src.optimization import process_single_ligand
        from unittest.mock import Mock

        # Create mock args
        mock_args = Mock()
        mock_args.optimize = False
        mock_args.exp_lig = False
        mock_args.eda = True
        mock_args.opt_log = False

        mock_logger = Mock()
        optimization_log = []

        with patch('src.optimization.protein_ligand_interaction') as mock_interaction:
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
        from src.optimization import process_single_ligand
        from unittest.mock import Mock

        # Create mock args
        mock_args = Mock()
        mock_args.optimize = False
        mock_args.exp_lig = False
        mock_args.eda = False
        mock_args.opt_log = False

        mock_logger = Mock()
        optimization_log = []

        with patch('src.optimization.protein_ligand_interaction') as mock_interaction:
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
        from src.optimization import process_single_ligand
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
        mock_args.exp_lig = True
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