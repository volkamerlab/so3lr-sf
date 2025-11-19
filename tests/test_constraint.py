"""
Tests for the constraint module.
"""

import pytest
import numpy as np
import logging
from unittest.mock import Mock, patch

from src.constraint import (
    _constrain_by_atoms,
    _constrain_by_residues,
    create_optimization_constraint
)
from src.molecule_loader import load_ase_structure


class TestConstraints:
    """Tests for constraint functionality."""

    @pytest.mark.unit
    def test_constrain_by_atoms_function(self, water_files, alanine_files):
        """Test _constrain_by_atoms function directly."""
        import logging

        # Load test structures
        protein = load_ase_structure(alanine_files['xyz'])[0]
        ligand = load_ase_structure(water_files['xyz'])[0]
        complex_atoms = protein + ligand

        logger = logging.getLogger(__name__)

        # Test with very small radius (should fix many atoms)
        fixed_atoms_small = _constrain_by_atoms(complex_atoms, len(protein), 0.5, logger)

        # Test with large radius (should fix fewer/no atoms)
        fixed_atoms_large = _constrain_by_atoms(complex_atoms, len(protein), 10.0, logger)

        # Small radius should fix more atoms than large radius
        assert len(fixed_atoms_small) >= len(fixed_atoms_large)

        # All indices should be valid protein atom indices
        for idx in fixed_atoms_small:
            assert 0 <= idx < len(protein)
        for idx in fixed_atoms_large:
            assert 0 <= idx < len(protein)

    @pytest.mark.unit
    def test_constrain_by_residues_function(self, water_files, alanine_files):
        """Test _constrain_by_residues function directly."""
        import logging

        # Use PDB file for residue information
        protein_path = alanine_files['pdb']
        protein = load_ase_structure(protein_path)[0]
        ligand = load_ase_structure(water_files['xyz'])[0]
        complex_atoms = protein + ligand

        logger = logging.getLogger(__name__)

        # Test residue-based constraints
        fixed_atoms = _constrain_by_residues(complex_atoms, len(protein), 2.0, protein_path, logger)

        # Should return valid atom indices
        assert isinstance(fixed_atoms, list)
        assert all(isinstance(idx, int) for idx in fixed_atoms)
        assert all(0 <= idx < len(protein) for idx in fixed_atoms)

        # Indices should be sorted
        assert fixed_atoms == sorted(fixed_atoms)

    @pytest.mark.unit
    def test_constrain_by_residues_fallback(self, water_files, alanine_files, caplog):
        """Test _constrain_by_residues falls back to atom-based on error."""
        import logging

        # Use XYZ file which should cause residue parsing to fail
        protein_path = alanine_files['xyz']
        protein = load_ase_structure(protein_path)[0]
        ligand = load_ase_structure(water_files['xyz'])[0]
        complex_atoms = protein + ligand

        logger = logging.getLogger(__name__)

        with caplog.at_level(logging.WARNING):
            fixed_atoms = _constrain_by_residues(complex_atoms, len(protein), 2.0, protein_path, logger)

        # Should still return valid results from fallback
        assert isinstance(fixed_atoms, list)
        assert all(0 <= idx < len(protein) for idx in fixed_atoms)

        # Should have logged fallback warnings
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("Residue-based constraints failed" in msg for msg in warning_messages)
        assert any("Falling back to atom-based constraints" in msg for msg in warning_messages)

    @pytest.mark.unit
    def test_create_optimization_constraint_pdb_format(self, water_files, alanine_files, caplog):
        """Test create_optimization_constraint with PDB format (residue-level)."""
        import logging

        protein_path = alanine_files['pdb']
        protein = load_ase_structure(protein_path)[0]
        ligand = load_ase_structure(water_files['xyz'])[0]
        complex_atoms = protein + ligand

        with caplog.at_level(logging.INFO):
            constraint = create_optimization_constraint(
                complex_atoms, len(protein), 2.0, protein_path
            )

        # Check info messages were logged
        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("Using residue-level constraints for PDB file" in msg for msg in info_messages)
        assert any("Complete residues will be flexible" in msg for msg in info_messages)

        # Constraint could be None (if all atoms are flexible) or FixAtoms object
        if constraint is not None:
            from ase.constraints import FixAtoms
            assert isinstance(constraint, FixAtoms)
            assert all(0 <= idx < len(protein) for idx in constraint.indices)

    @pytest.mark.unit
    def test_create_optimization_constraint_xyz_format(self, water_files, alanine_files, caplog):
        """Test create_optimization_constraint with XYZ format (atom-level)."""
        import logging

        protein_path = alanine_files['xyz']
        protein = load_ase_structure(protein_path)[0]
        ligand = load_ase_structure(water_files['xyz'])[0]
        complex_atoms = protein + ligand

        with caplog.at_level(logging.WARNING):
            constraint = create_optimization_constraint(
                complex_atoms, len(protein), 2.0, protein_path
            )

        # Check warning messages were logged
        warning_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("Using atom-level constraints" in msg for msg in warning_messages)
        assert any("residues may be split" in msg for msg in warning_messages)
        assert any("use PDB format input files" in msg for msg in warning_messages)

        # Constraint could be None or FixAtoms object
        if constraint is not None:
            from ase.constraints import FixAtoms
            assert isinstance(constraint, FixAtoms)

    @pytest.mark.unit
    def test_create_optimization_constraint_no_protein_path(self, water_files, alanine_files, caplog):
        """Test create_optimization_constraint with no protein path (defaults to atom-level)."""
        import logging

        protein = load_ase_structure(alanine_files['xyz'])[0]
        ligand = load_ase_structure(water_files['xyz'])[0]
        complex_atoms = protein + ligand

        with caplog.at_level(logging.INFO):
            constraint = create_optimization_constraint(
                complex_atoms, len(protein), 2.0, None
            )

        # Check default message was logged
        info_messages = [r.message for r in caplog.records if r.levelno == logging.INFO]
        assert any("No protein path provided - using atom-level constraints" in msg for msg in info_messages)

    @pytest.mark.unit
    def test_create_optimization_constraint_with_fixed_atoms(self, sample_complex_atoms):
        """Test constraint creation when some atoms should be fixed."""

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

    @pytest.mark.integration
    @pytest.mark.slow
    def test_create_optimization_constraint_real(self, test_data_dir, temp_dir):
        """Test create_optimization_constraint with real complex atoms."""
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