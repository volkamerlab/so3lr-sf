"""
Tests for the trim module.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from src.trim import (
    trim_structure,
    perform_trimming,
    _trim_by_atoms,
    _trim_by_residues,
    _bridge_short_gaps,
    _saturate_open_valences,
    MAX_BRIDGE_GAP,
)
from src.molecule_loader import load_ase_structure, prepare_mda_universe


class TestTrimStructure:
    """Tests for structure trimming functionality."""

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
    def test_trim_structure_pdb_residue_based(self, water_files, alanine_files, temp_dir):
        """Test residue-based trimming for PDB files."""
        protein_path = alanine_files['pdb']  # Use PDB format
        ligand_path = water_files['xyz']
        radius = 2.0

        trimmed_protein_path = trim_structure(
            protein_path, ligand_path, radius=radius, output_dir=temp_dir
        )

        # Check that files were created with residue suffix
        expected_filename = f"{protein_path.stem}_trimmed_{radius}A_residue.pdb"
        assert Path(trimmed_protein_path).name == expected_filename
        assert Path(trimmed_protein_path).exists()

        # Check that trimmed protein is valid
        original_protein = load_ase_structure(protein_path)[0]
        trimmed_protein = load_ase_structure(trimmed_protein_path)[0]

        assert len(trimmed_protein) <= len(original_protein)
        assert len(trimmed_protein) > 0

    @pytest.mark.unit
    def test_trim_structure_xyz_atom_based_warnings(self, water_files, alanine_files, temp_dir, caplog):
        """Test that XYZ files generate appropriate warnings for atom-based trimming."""
        import logging
        protein_path = alanine_files['xyz']  # Use XYZ format
        ligand_path = water_files['xyz']

        with caplog.at_level(logging.WARNING):
            trimmed_protein_path = trim_structure(
                protein_path, ligand_path, radius=2.0, output_dir=temp_dir
            )

        # Check warning messages were logged
        warning_messages = [record.message for record in caplog.records if record.levelno >= logging.WARNING]
        assert any("Using atom-based trimming" in msg for msg in warning_messages)
        assert any("NOT guaranteed" in msg for msg in warning_messages)
        assert any("Use PDB input" in msg for msg in warning_messages)

        # Check filename has atom suffix
        expected_filename = f"{protein_path.stem}_trimmed_2.0A_atom.xyz"
        assert Path(trimmed_protein_path).name == expected_filename

    @pytest.mark.unit
    def test_trim_structure_small_radius(self, water_files, alanine_files, temp_dir):
        """Test trimming with very small radius."""
        protein_path = alanine_files['xyz']
        ligand_path = water_files['xyz']

        # Use unrealistically small radius
        with pytest.raises(ValueError, match="No protein atoms found within"):
            trim_structure(protein_path, ligand_path, radius=0.1, output_dir=temp_dir)

    @pytest.mark.unit
    def test_trim_by_atoms_function(self, water_files, alanine_files):
        """Test _trim_by_atoms function directly."""
        import logging

        # Load test structures
        protein = load_ase_structure(alanine_files['xyz'])[0]
        ligand = load_ase_structure(water_files['xyz'])[0]

        protein_positions = protein.get_positions()
        ligand_positions = ligand.get_positions()
        logger = logging.getLogger(__name__)

        # Test with small radius
        atoms_to_keep_small = _trim_by_atoms(protein_positions, ligand_positions, 1.0, logger)

        # Test with large radius
        atoms_to_keep_large = _trim_by_atoms(protein_positions, ligand_positions, 10.0, logger)

        # Large radius should keep more or equal atoms than small radius
        assert len(atoms_to_keep_large) >= len(atoms_to_keep_small)

        # All atom indices should be valid
        for idx in atoms_to_keep_small:
            assert 0 <= idx < len(protein_positions)
        for idx in atoms_to_keep_large:
            assert 0 <= idx < len(protein_positions)

    @pytest.mark.unit
    def test_trim_by_residues_function(self, water_files, alanine_files):
        """Test _trim_by_residues function directly."""
        import logging

        # Use PDB file for residue information
        protein_path = alanine_files['pdb']
        protein = load_ase_structure(protein_path)[0]
        ligand = load_ase_structure(water_files['xyz'])[0]

        protein_positions = protein.get_positions()
        ligand_positions = ligand.get_positions()
        logger = logging.getLogger(__name__)

        # Test residue-based trimming
        atoms_to_keep, caps = _trim_by_residues(
            protein_path, protein, protein_positions, ligand_positions, 3.0, logger
        )

        # Should return valid atom indices
        assert isinstance(atoms_to_keep, list)
        assert len(atoms_to_keep) > 0
        assert len(atoms_to_keep) <= len(protein_positions)

        # All indices should be valid
        for idx in atoms_to_keep:
            assert 0 <= idx < len(protein_positions)

        # Indices should be sorted
        assert atoms_to_keep == sorted(atoms_to_keep)

        # Caps are (symbol, position) tuples
        assert isinstance(caps, list)
        for symbol, position in caps:
            assert symbol == "H"
            assert len(position) == 3

    @pytest.mark.unit
    def test_trim_by_residues_fallback_to_atoms(self, water_files, alanine_files, caplog):
        """Test that _trim_by_residues falls back to atom-based trimming on error."""
        import logging

        # Use XYZ file which should cause residue parsing to fail
        protein_path = alanine_files['xyz']  # XYZ instead of PDB
        protein = load_ase_structure(protein_path)[0]
        ligand = load_ase_structure(water_files['xyz'])[0]

        protein_positions = protein.get_positions()
        ligand_positions = ligand.get_positions()
        logger = logging.getLogger(__name__)

        with caplog.at_level(logging.WARNING):
            atoms_to_keep, caps = _trim_by_residues(
                protein_path, protein, protein_positions, ligand_positions, 3.0, logger
            )

        # Should still return valid results (from fallback)
        assert isinstance(atoms_to_keep, list)
        assert len(atoms_to_keep) > 0
        assert isinstance(caps, list)

        # Should have logged fallback warnings
        warning_messages = [record.message for record in caplog.records if record.levelno >= logging.WARNING]
        assert any("Residue-based trimming failed" in msg for msg in warning_messages)
        assert any("Falling back to atom-based trimming" in msg for msg in warning_messages)

    @pytest.mark.unit
    def test_perform_trimming_with_specified_ligand(self, temp_dir, sample_xyz_file, sample_sdf_file, mock_logger):
        """Test perform_trimming with specified trim ligand."""
        with patch('src.trim.trim_structure') as mock_trim:
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
    def test_perform_trimming_no_ligand_files_found(self, temp_dir, sample_xyz_file, mock_logger):
        """Test perform_trimming when no ligand files are found."""
        with patch('src.trim.get_ligand_files', return_value=[]):
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
        with patch('src.trim.get_ligand_files', return_value=[sample_sdf_file]), \
             patch('src.trim.trim_structure') as mock_trim:

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

    @pytest.mark.unit
    def test_trim_structure_unsupported_format(self, water_files, temp_dir):
        """Test trimming with SDF format (should raise error for unsupported format)."""
        protein_path = water_files['sdf']  # Use SDF as "protein"
        ligand_path = water_files['xyz']

        # SDF format is not supported for protein input
        with pytest.raises(ValueError, match="Unsupported protein file format for trimming: .sdf"):
            trim_structure(protein_path, ligand_path, radius=2.0, output_dir=temp_dir)

    @pytest.mark.unit
    def test_trim_structure_large_radius_includes_all(self, water_files, alanine_files, temp_dir):
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


class TestGapBridging:
    """Tests for short-sequence-gap bridging in residue-based trimming."""

    @pytest.mark.unit
    def test_bridge_short_gaps_fills_gap_within_threshold(self):
        """A gap no larger than MAX_BRIDGE_GAP is filled."""
        # Linear chain 0-1-2-...-9
        next_of = {i: i + 1 for i in range(9)}
        prev_of = {i + 1: i for i in range(9)}

        # Selected 2 and 5 -> gap is residues 3,4 (size 2)
        bridged = _bridge_short_gaps({2, 5}, next_of, prev_of, 10, max_gap=2)
        assert bridged == {3, 4}

    @pytest.mark.unit
    def test_bridge_short_gaps_leaves_long_gap_alone(self):
        """A gap larger than max_gap is left as a genuine break."""
        next_of = {i: i + 1 for i in range(9)}
        prev_of = {i + 1: i for i in range(9)}

        # Selected 0 and 4 -> gap is residues 1,2,3 (size 3) > max_gap 2
        bridged = _bridge_short_gaps({0, 4}, next_of, prev_of, 10, max_gap=2)
        assert bridged == set()

    @pytest.mark.unit
    def test_bridge_short_gaps_disabled(self):
        """max_gap of 0 disables bridging."""
        next_of = {i: i + 1 for i in range(9)}
        prev_of = {i + 1: i for i in range(9)}

        assert _bridge_short_gaps({2, 5}, next_of, prev_of, 10, max_gap=0) == set()

    @pytest.mark.unit
    def test_bridge_short_gaps_does_not_cross_chain_break(self):
        """Residues on different chains are never bridged."""
        # Two separate chains: 0-1-2 and 3-4-5, no link between 2 and 3
        next_of = {0: 1, 1: 2, 3: 4, 4: 5}
        prev_of = {1: 0, 2: 1, 4: 3, 5: 4}

        bridged = _bridge_short_gaps({2, 3}, next_of, prev_of, 6, max_gap=2)
        assert bridged == set()

    @pytest.mark.unit
    def test_short_gap_bridged_end_to_end(self, peptide_files, caplog):
        """Isolated single-residue gaps in the pocket are filled back in."""
        import logging

        protein = load_ase_structure(peptide_files['pdb'])[0]
        ligand = load_ase_structure(peptide_files['ligand'])[0]

        with caplog.at_level(logging.INFO):
            atoms_to_keep, _ = _trim_by_residues(
                peptide_files['pdb'], protein, protein.get_positions(),
                ligand.get_positions(), 4.0, logging.getLogger(__name__)
            )

        universe = prepare_mda_universe(peptide_files['pdb'])
        resid_by_atom = {int(a.index): int(a.residue.resid) for a in universe.atoms}
        kept_resids = sorted({resid_by_atom[i] for i in atoms_to_keep})

        # The kept residues form one contiguous run (no gaps left)
        assert kept_resids == list(range(kept_resids[0], kept_resids[-1] + 1))
        assert any("Bridged" in r.message for r in caplog.records)

    @pytest.mark.unit
    def test_large_gap_not_bridged_end_to_end(self, peptide_files):
        """A multi-residue gap between two pocket segments is preserved."""
        import logging

        protein = load_ase_structure(peptide_files['pdb'])[0]
        ligand = load_ase_structure(peptide_files['ligand'])[0]

        atoms_to_keep, _ = _trim_by_residues(
            peptide_files['pdb'], protein, protein.get_positions(),
            ligand.get_positions(), 3.0, logging.getLogger(__name__)
        )

        universe = prepare_mda_universe(peptide_files['pdb'])
        resid_by_atom = {int(a.index): int(a.residue.resid) for a in universe.atoms}
        kept_resids = sorted({resid_by_atom[i] for i in atoms_to_keep})

        # Two distinct segments remain -> there is at least one internal gap
        assert kept_resids != list(range(kept_resids[0], kept_resids[-1] + 1))


class TestValenceCapping:
    """Tests for hydrogen-capping of valences severed by trimming."""

    @pytest.mark.unit
    def test_caps_added_at_truncation_boundary(self, peptide_files):
        """Trimming an internal pocket adds capping hydrogens, all element H."""
        import logging

        protein = load_ase_structure(peptide_files['pdb'])[0]
        ligand = load_ase_structure(peptide_files['ligand'])[0]

        atoms_to_keep, caps = _trim_by_residues(
            peptide_files['pdb'], protein, protein.get_positions(),
            ligand.get_positions(), 3.0, logging.getLogger(__name__)
        )

        assert len(caps) > 0
        assert all(symbol == "H" for symbol, _ in caps)

        # Every cap sits at a plausible X-H bond length from some kept atom
        kept_positions = protein.get_positions()[atoms_to_keep]
        for _, cap_pos in caps:
            nearest = float(np.linalg.norm(kept_positions - cap_pos, axis=1).min())
            assert 0.8 <= nearest <= 1.4

    @pytest.mark.unit
    def test_capped_hydrogens_survive_the_write(self, peptide_files, temp_dir):
        """The written trimmed PDB has exactly (kept atoms + caps), caps last and all H."""
        import logging
        from src.utils import write_structure

        protein = load_ase_structure(peptide_files['pdb'])[0]
        ligand = load_ase_structure(peptide_files['ligand'])[0]

        atoms_to_keep, caps = _trim_by_residues(
            peptide_files['pdb'], protein, protein.get_positions(),
            ligand.get_positions(), 3.0, logging.getLogger(__name__)
        )
        assert len(caps) > 0

        # Copy the fixture into temp_dir so the trimmed file is not written next
        # to the tracked fixture (trim_structure ignores output_dir).
        pdb_copy = temp_dir / "peptide.pdb"
        write_structure(protein, pdb_copy)
        trimmed = load_ase_structure(
            trim_structure(pdb_copy, peptide_files['ligand'], radius=3.0)
        )[0]

        assert len(trimmed) == len(atoms_to_keep) + len(caps)
        cap_symbols = trimmed.get_chemical_symbols()[len(atoms_to_keep):]
        assert all(s == "H" for s in cap_symbols)

    @pytest.mark.unit
    def test_no_spurious_caps_from_placeholder_cell(self, peptide_files):
        """A placeholder CRYST1 cell must not make bond perception wrap on images."""
        import logging

        protein = load_ase_structure(peptide_files['pdb'])[0]
        ligand = load_ase_structure(peptide_files['ligand'])[0]

        # Precondition: the fixture actually carries the placeholder cell that
        # would trigger the bug, otherwise this test proves nothing.
        assert protein.get_cell().volume < 10.0

        _, caps = _trim_by_residues(
            peptide_files['pdb'], protein, protein.get_positions(),
            ligand.get_positions(), 3.0, logging.getLogger(__name__)
        )
        # A ~200-atom fragment has a handful of boundary bonds, not hundreds
        assert len(caps) < 20

    @pytest.mark.unit
    def test_xyz_path_warns_and_does_not_cap(self, peptide_files, temp_dir, caplog):
        """XYZ input is not capped (no topology) but warns loudly about it."""
        import logging
        from src.utils import write_structure

        # Produce an XYZ copy of the peptide to trim
        peptide = load_ase_structure(peptide_files['pdb'])[0]
        xyz_path = temp_dir / "peptide.xyz"
        write_structure(peptide, xyz_path)

        with caplog.at_level(logging.WARNING):
            trimmed_path = trim_structure(
                xyz_path, peptide_files['ligand'], radius=3.0, output_dir=temp_dir
            )

        trimmed = load_ase_structure(trimmed_path)[0]
        # Atom-based result equals the raw within-radius selection, nothing appended
        raw = _trim_by_atoms(
            peptide.get_positions(),
            load_ase_structure(peptide_files['ligand'])[0].get_positions(),
            3.0, logging.getLogger(__name__),
        )
        assert len(trimmed) == len(raw)

        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("NOT guaranteed" in m for m in warnings)
        assert any("Use PDB input" in m for m in warnings)

    @pytest.mark.unit
    def test_residue_logic_failure_propagates(self, peptide_files, monkeypatch):
        """A bug after topology load must raise, not silently fall back to atoms."""
        import logging
        import src.trim as trim_module

        protein = load_ase_structure(peptide_files['pdb'])[0]
        ligand = load_ase_structure(peptide_files['ligand'])[0]

        def boom(*args, **kwargs):
            raise RuntimeError("injected failure in peptide graph")

        monkeypatch.setattr(trim_module, "_build_peptide_graph", boom)

        with pytest.raises(RuntimeError, match="injected failure"):
            _trim_by_residues(
                peptide_files['pdb'], protein, protein.get_positions(),
                ligand.get_positions(), 3.0, logging.getLogger(__name__)
            )