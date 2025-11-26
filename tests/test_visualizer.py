"""
Tests for 3D protein-ligand energy visualizer module.

This module tests PyMOL script generation functionality including:
- Interaction detection commands
- Component title creation
- Energy-based coloring
- Complete PyMOL session creation
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import mock_open, patch
from io import StringIO

from src.visualizer import (
    add_interaction_detection,
    add_component_title,
    apply_energy_coloring,
    create_pymol_session
)


class TestAddInteractionDetection:
    """Test the add_interaction_detection function that generates PyMOL interaction commands."""

    def test_add_interaction_detection_basic(self):
        """Test that basic interaction detection commands are generated correctly."""
        output = StringIO()
        structure_name = "test_protein"

        add_interaction_detection(output, structure_name)

        commands = output.getvalue()

        # Check for header comment
        assert "# Detect and visualize interactions between ligand and protein" in commands

        # Check hydrogen bond detection
        assert "# Hydrogen bonds (3.2 Å cutoff)" in commands
        assert f"select hbonds_{structure_name}" in commands
        assert f"distance hbond_dist_{structure_name}" in commands
        assert "all within 3.2 of (organic and (elem N+O+S))" in commands

        # Check hydrophobic contact detection
        assert "# Hydrophobic contacts (4.5 Å cutoff for carbon atoms)" in commands
        assert f"select hydrophobic_{structure_name}" in commands
        assert "all within 4.5 of (organic and elem C)" in commands

        # Check ionic interaction detection
        assert "# Salt bridges/ionic interactions (4.0 Å cutoff)" in commands
        assert f"select ionic_{structure_name}" in commands
        assert "resn ARG+LYS+HIS and name NH*+NZ+ND1+NE2" in commands
        assert "resn ASP+GLU and name OD*+OE*" in commands
        assert "all within 4.0 of organic" in commands

        # Check aromatic interaction detection
        assert "# Pi-pi and cation-pi interactions (aromatic rings, 5.0 Å cutoff)" in commands
        assert f"select aromatic_{structure_name}" in commands
        assert "resn PHE and name CG+CD*+CE*+CZ" in commands
        assert "resn TYR and name CG+CD*+CE*+CZ+OH" in commands
        assert "resn TRP and name CG+CD*+NE1+CE*+CZ*+CH2" in commands
        assert "resn HIS and name CG+ND1+CD2+CE1+NE2" in commands
        assert "all within 5.0 of organic" in commands

    def test_add_interaction_detection_styling(self):
        """Test that styling commands for interactions are generated correctly."""
        output = StringIO()
        structure_name = "test_protein"

        add_interaction_detection(output, structure_name)

        commands = output.getvalue()

        # Check styling commands
        assert "# Style interaction selections - show whole residues as sticks" in commands
        assert f"show sticks, {structure_name} and polymer and (byres hbonds_{structure_name})" in commands
        assert f"show sticks, {structure_name} and polymer and (byres hydrophobic_{structure_name})" in commands
        assert f"show sticks, {structure_name} and polymer and (byres ionic_{structure_name})" in commands
        assert f"show sticks, {structure_name} and polymer and (byres aromatic_{structure_name})" in commands

        # Check distance label styling
        assert f"hide labels, hbond_dist_{structure_name}" in commands
        assert f"color yellow, hbond_dist_{structure_name}" in commands
        assert f"set dash_width, 2, hbond_dist_{structure_name}" in commands

    def test_add_interaction_detection_structure_specific(self):
        """Test that structure-specific names are used correctly to avoid conflicts."""
        output = StringIO()
        structure_name = "protein_mlff"

        add_interaction_detection(output, structure_name)

        commands = output.getvalue()

        # Verify all selections use the structure-specific name
        assert f"hbonds_{structure_name}" in commands
        assert f"hydrophobic_{structure_name}" in commands
        assert f"ionic_{structure_name}" in commands
        assert f"aromatic_{structure_name}" in commands
        assert f"hbond_dist_{structure_name}" in commands

        # Count occurrences to ensure all instances use structure name
        assert commands.count(structure_name) >= 15  # Should appear in multiple commands


class TestAddComponentTitle:
    """Test the add_component_title function that creates floating labels."""

    def test_add_component_title_positioning(self):
        """Test that component titles are positioned correctly above structures."""
        output = StringIO()
        structure_name = "protein_mlff"
        component = "MLFF"
        center_of_mass = np.array([10.0, 20.0, 30.0])
        x_offset = 50.0

        add_component_title(output, structure_name, component, center_of_mass, x_offset)

        commands = output.getvalue()

        # Check title positioning calculation
        expected_x = center_of_mass[0] + x_offset  # 60.0
        expected_y = center_of_mass[1]             # 20.0
        expected_z = center_of_mass[2] + 50        # 80.0

        assert f"pos=[{expected_x:.3f}, {expected_y:.3f}, {expected_z:.3f}]" in commands
        assert f"pseudoatom title_{structure_name}" in commands

    def test_add_component_title_styling(self):
        """Test that component title styling is applied correctly."""
        output = StringIO()
        structure_name = "protein_electrostatics"
        component = "Electrostatics"
        center_of_mass = np.array([0.0, 0.0, 0.0])
        x_offset = 0.0

        add_component_title(output, structure_name, component, center_of_mass, x_offset)

        commands = output.getvalue()

        # Check header comment
        assert f"# Add floating title for {component} energy component at structure center" in commands

        # Check label content and styling
        assert f"label title_{structure_name}, '{component}'" in commands
        assert f"set label_size, 25, title_{structure_name}" in commands
        assert f"set label_color, black, title_{structure_name}" in commands

        # Check pseudoatom styling
        assert f"show spheres, title_{structure_name}" in commands
        assert f"set sphere_scale, 0.2, title_{structure_name}" in commands
        assert f"color blue, title_{structure_name}" in commands

    def test_add_component_title_multiple_components(self):
        """Test that titles work correctly for multiple components side by side."""
        components = [("MLFF", 0.0), ("Electrostatics", 70.0), ("Total", 140.0)]
        center_of_mass = np.array([25.0, 15.0, 10.0])

        outputs = []
        for component, x_offset in components:
            output = StringIO()
            structure_name = f"protein_{component.lower()}"
            add_component_title(output, structure_name, component, center_of_mass, x_offset)
            outputs.append(output.getvalue())

        # Verify each component has unique positioning
        expected_positions = [
            "[25.000, 15.000, 60.000]",  # MLFF: 25+0, 15, 10+50
            "[95.000, 15.000, 60.000]",  # Electrostatics: 25+70, 15, 10+50
            "[165.000, 15.000, 60.000]"  # Total: 25+140, 15, 10+50
        ]

        for i, (output, expected_pos) in enumerate(zip(outputs, expected_positions)):
            assert expected_pos in output
            assert f"'{components[i][0]}'" in output


class TestApplyEnergyColoring:
    """Test the apply_energy_coloring function that creates energy-based gradients."""

    def test_apply_energy_coloring_threshold_filtering(self):
        """Test that only significant energy atoms (>5% threshold) are colored."""
        output = StringIO()
        structure_name = "test_protein"
        component = "MLFF"

        # Energy values: max_abs = 1.0, threshold = 0.05
        atom_weights = {
            "1": 1.0,    # 100% - should be colored (red)
            "2": -0.8,   # 80% - should be colored (blue)
            "3": 0.1,    # 10% - should be colored (red)
            "4": 0.02,   # 2% - should NOT be colored (below threshold)
            "5": -0.01   # 1% - should NOT be colored (below threshold)
        }

        contributing_atoms = apply_energy_coloring(output, structure_name, component, atom_weights)

        commands = output.getvalue()

        # Check that only significant atoms are colored
        assert len(contributing_atoms) == 3
        assert set(contributing_atoms) == {1, 2, 3}

        # Check color definitions for significant atoms
        assert "red_MLFF_0" in commands  # atom 1 (positive)
        assert "blue_MLFF_1" in commands  # atom 2 (negative)
        assert "red_MLFF_2" in commands  # atom 3 (positive)

        # Check that atoms 4 and 5 are not colored
        assert "id 4" not in commands
        assert "id 5" not in commands

        # Check threshold message
        assert "Colored 3 atoms with significant energy (>0.0500)" in commands

    def test_apply_energy_coloring_gradient_calculation(self):
        """Test that color gradients are calculated correctly."""
        output = StringIO()
        structure_name = "test_protein"
        component = "TEST"

        # Simple energy values for testing
        atom_weights = {
            "1": 1.0,   # Maximum positive (100% intensity)
            "2": -1.0,  # Maximum negative (100% intensity)
            "3": 0.5,   # 50% positive intensity
            "4": -0.5   # 50% negative intensity
        }

        apply_energy_coloring(output, structure_name, component, atom_weights)

        commands = output.getvalue()

        # For maximum positive (intensity = 1.0): r = 0.7 + 0.3*1 = 1.0, g = b = 0.7*0 = 0.0
        assert "set_color red_TEST_0, [1.000, 0.000, 0.000]" in commands

        # For maximum negative (intensity = 1.0): r = g = 0.7*0 = 0.0, b = 0.7 + 0.3*1 = 1.0
        assert "set_color blue_TEST_1, [0.000, 0.000, 1.000]" in commands

        # For 50% positive (intensity = 0.5): r = 0.7 + 0.3*0.5 = 0.85, g = b = 0.7*0.5 = 0.35
        assert "set_color red_TEST_2, [0.850, 0.350, 0.350]" in commands

        # For 50% negative (intensity = 0.5): r = g = 0.7*0.5 = 0.35, b = 0.7 + 0.3*0.5 = 0.85
        assert "set_color blue_TEST_3, [0.350, 0.350, 0.850]" in commands

    def test_apply_energy_coloring_invalid_values(self, caplog):
        """Test handling of invalid energy values."""
        output = StringIO()
        structure_name = "test_protein"
        component = "TEST"

        atom_weights = {
            "1": 1.0,        # Valid
            "2": "invalid",  # Invalid string
            "3": None,       # Invalid None
            "4": -0.5        # Valid
        }

        contributing_atoms = apply_energy_coloring(output, structure_name, component, atom_weights)

        # Should only include valid atoms (1 and 4)
        assert len(contributing_atoms) == 2
        assert set(contributing_atoms) == {1, 4}

        # Check that warnings were logged for invalid values
        assert "Invalid energy value at atom 2" in caplog.text
        assert "Invalid energy value at atom 3" in caplog.text

    def test_apply_energy_coloring_empty_weights(self):
        """Test behavior with empty atom weights."""
        output = StringIO()
        structure_name = "test_protein"
        component = "TEST"
        atom_weights = {}

        # This should raise ValueError due to empty sequence
        with pytest.raises(ValueError):
            apply_energy_coloring(output, structure_name, component, atom_weights)


class TestCreatePymolSession:
    """Test the complete PyMOL session creation function."""

    def test_create_pymol_session_file_validation(self):
        """Test that file validation works correctly."""
        # Test with non-existent file
        with pytest.raises(FileNotFoundError, match="Protein file not found"):
            create_pymol_session(
                "/nonexistent/file.pdb",
                {"MLFF": {"1": 0.5}},
                "/tmp/output.pml",
                np.array([0, 0, 0])
            )

    def test_create_pymol_session_basic_structure(self):
        """Test basic PyMOL session script structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a temporary PDB file
            pdb_file = Path(temp_dir) / "test.pdb"
            pdb_file.write_text("ATOM      1  N   ALA A   1      20.154  17.512  11.096  1.00 20.00           N\n")

            output_file = Path(temp_dir) / "output.pml"
            atom_weights = {
                "MLFF": {"1": 0.5, "2": -0.3},
                "Electrostatics": {"1": -0.2, "2": 0.8}
            }
            center_of_mass = np.array([0, 0, 0])

            result = create_pymol_session(pdb_file, atom_weights, output_file, center_of_mass)

            assert result == str(output_file)
            assert output_file.exists()

            content = output_file.read_text()

            # Check header
            assert "# PyMOL Protein Energy Visualization" in content
            assert f"# Generated from: {pdb_file}" in content
            assert "# Components: MLFF, Electrostatics" in content

            # Check basic setup
            assert "delete all" in content
            assert "bg_color white" in content

            # Check that structures are loaded for each component
            assert "load test.pdb, protein_mlff" in content
            assert "load test.pdb, protein_electrostatics" in content

            # Check structure positioning (overlapping - no translations)
            assert "translate [" not in content

    def test_create_pymol_session_protein_ligand_styling(self):
        """Test that protein and ligand styling commands are generated."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pdb_file = Path(temp_dir) / "complex.pdb"
            pdb_file.write_text("ATOM      1  N   ALA A   1      20.154  17.512  11.096  1.00 20.00           N\n")

            output_file = Path(temp_dir) / "output.pml"
            atom_weights = {"Total": {"1": 0.7}}
            center_of_mass = np.array([10, 20, 30])

            create_pymol_session(pdb_file, atom_weights, output_file, center_of_mass)

            content = output_file.read_text()

            # Check protein styling
            assert "show cartoon, protein_total and polymer" in content
            assert "color gray80, protein_total and polymer" in content

            # Check ligand styling
            assert "show sticks, protein_total and organic" in content
            assert "color gray60, protein_total and organic" in content

            # Check near ligand selection
            assert "select near_ligand_protein_total, protein_total and (all within 10 of organic)" in content
            assert "color gray70, near_ligand_protein_total" in content

    def test_create_pymol_session_energy_components(self):
        """Test that energy components are processed correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pdb_file = Path(temp_dir) / "test.pdb"
            pdb_file.write_text("ATOM      1  N   ALA A   1      20.154  17.512  11.096  1.00 20.00           N\n")

            output_file = Path(temp_dir) / "output.pml"
            atom_weights = {
                "MLFF": {"1": 1.0, "2": -0.5},
                "ZBL": {"1": 0.3, "2": 0.8},
                "Dispersion": {"1": -0.2, "2": -0.1}
            }
            center_of_mass = np.array([5, 10, 15])

            create_pymol_session(pdb_file, atom_weights, output_file, center_of_mass)

            content = output_file.read_text()

            # Check that all components are processed
            assert "protein_mlff" in content
            assert "protein_zbl" in content
            assert "protein_dispersion" in content

            # Check that structures are positioned at same location (no translation)
            assert "translate [" not in content

            # Check interaction detection is called only once (not for each component)
            assert content.count("# Detect and visualize interactions") == 1

            # Check that near_ligand selections are cleaned up
            assert "delete near_ligand_protein_mlff" in content
            assert "delete near_ligand_protein_zbl" in content
            assert "delete near_ligand_protein_dispersion" in content

    def test_create_pymol_session_final_setup(self):
        """Test that final visualization setup commands are included."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pdb_file = Path(temp_dir) / "test.pdb"
            pdb_file.write_text("ATOM      1  N   ALA A   1      20.154  17.512  11.096  1.00 20.00           N\n")

            output_file = Path(temp_dir) / "output.pml"
            atom_weights = {"Test": {"1": 0.5}}
            center_of_mass = np.array([0, 0, 0])

            create_pymol_session(pdb_file, atom_weights, output_file, center_of_mass)

            content = output_file.read_text()

            # Check final setup commands
            assert "# Final visualization setup" in content
            assert "zoom all" in content
            assert "set ray_opaque_background, off" in content
            assert "set ray_trace_mode, 1" in content
            assert "orient" in content

            # Check color scale information
            assert "# Color scale information:" in content
            assert "# Blue = Low energy values" in content
            assert "# Red = High energy values" in content

            # Check session saving
            assert "# Save session" in content
            assert "save output.pse" in content

    def test_create_pymol_session_contributing_atoms_display(self):
        """Test that contributing atoms are displayed as sticks correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pdb_file = Path(temp_dir) / "test.pdb"
            pdb_file.write_text("ATOM      1  N   ALA A   1      20.154  17.512  11.096  1.00 20.00           N\n")

            output_file = Path(temp_dir) / "output.pml"
            # High energy values to ensure they exceed threshold
            atom_weights = {"MLFF": {"10": 2.0, "20": -1.5, "30": 1.0}}
            center_of_mass = np.array([0, 0, 0])

            create_pymol_session(pdb_file, atom_weights, output_file, center_of_mass)

            content = output_file.read_text()

            # Check that contributing atoms are shown as sticks
            assert "# Show sticks for whole residues containing contributing atoms" in content
            # The exact atom IDs will depend on apply_energy_coloring, but there should be a show sticks command
            assert "show sticks, protein_mlff and polymer and (byres (id" in content

    @patch('src.visualizer.logger')
    def test_create_pymol_session_logging(self, mock_logger):
        """Test that appropriate logging messages are generated."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pdb_file = Path(temp_dir) / "test.pdb"
            pdb_file.write_text("ATOM      1  N   ALA A   1      20.154  17.512  11.096  1.00 20.00           N\n")

            output_file = Path(temp_dir) / "output.pml"
            atom_weights = {"MLFF": {"1": 0.5}, "Electrostatics": {"1": -0.3}}
            center_of_mass = np.array([0, 0, 0])

            create_pymol_session(pdb_file, atom_weights, output_file, center_of_mass)

            # Check that info messages were logged
            mock_logger.info.assert_any_call("Creating PyMOL visualization with 2 energy components")
            mock_logger.info.assert_any_call(f"PyMOL script generated. To visualize: pymol {output_file}")


# Integration test to ensure all functions work together
class TestVisualizerIntegration:
    """Integration tests for the complete visualizer workflow."""

    def test_full_workflow_integration(self):
        """Test that all visualizer functions work together in a complete workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test PDB file
            pdb_content = """ATOM      1  N   MET A   1      20.154  17.512  11.096  1.00 20.00           N
ATOM      2  CA  MET A   1      19.030  16.739  10.498  1.00 20.00           C
ATOM      3  C   MET A   1      17.670  17.427  10.696  1.00 20.00           C
HETATM    4  C1  LIG B   1      15.123  18.456   9.234  1.00 20.00           C
HETATM    5  O1  LIG B   1      14.567  19.234   8.456  1.00 20.00           O
"""
            pdb_file = Path(temp_dir) / "complex.pdb"
            pdb_file.write_text(pdb_content)

            output_file = Path(temp_dir) / "visualization.pml"

            # Comprehensive energy data
            atom_weights = {
                "MLFF": {"1": 0.8, "2": -0.6, "3": 0.4, "4": -1.2, "5": 0.9},
                "Electrostatics": {"1": -0.3, "2": 0.7, "3": -0.5, "4": 1.1, "5": -0.8},
                "ZBL": {"1": 0.2, "2": 0.1, "3": 0.3, "4": 0.4, "5": 0.2},
                "Dispersion": {"1": -0.1, "2": -0.2, "3": -0.15, "4": -0.25, "5": -0.18},
                "Total": {"1": 0.6, "2": 0.0, "3": 0.05, "4": 0.25, "5": 0.12}
            }

            center_of_mass = np.array([17.5, 17.8, 10.4])

            # Run complete workflow
            result_path = create_pymol_session(pdb_file, atom_weights, output_file, center_of_mass)

            assert result_path == str(output_file)
            assert output_file.exists()

            content = output_file.read_text()

            # Verify all major sections are present
            sections_to_check = [
                "PyMOL Protein Energy Visualization",
                "delete all",
                "bg_color white",
                "load complex.pdb",
                "show cartoon",
                "show sticks",
                "set_color red_",
                "set_color blue_",
                "select hbonds_",
                "select hydrophobic_",
                "select ionic_",
                "select aromatic_",
                "zoom all",
                "save visualization.pse"
            ]

            for section in sections_to_check:
                assert section in content, f"Missing section: {section}"

            # Verify all energy components are processed
            for component in atom_weights.keys():
                structure_name = f"protein_{component.lower()}"
                assert structure_name in content

            # Verify overlapping positioning (no translations)
            assert "translate [" not in content

            # Verify interaction detection is called only once
            assert content.count("# Detect and visualize interactions") == 1

            # Verify near_ligand selections are cleaned up
            assert "delete near_ligand_protein_mlff" in content
            assert "delete near_ligand_protein_electrostatics" in content

            # Verify session file path
            session_file = output_file.with_suffix(".pse")
            assert f"save {session_file.name}" in content

    def test_edge_case_single_component(self):
        """Test workflow with single energy component."""
        with tempfile.TemporaryDirectory() as temp_dir:
            pdb_file = Path(temp_dir) / "single.pdb"
            pdb_file.write_text("ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00 20.00           N\n")

            output_file = Path(temp_dir) / "single.pml"
            atom_weights = {"Total": {"1": 1.0}}
            center_of_mass = np.array([0, 0, 0])

            result = create_pymol_session(pdb_file, atom_weights, output_file, center_of_mass)

            assert Path(result).exists()
            content = Path(result).read_text()

            # Should have single structure with no translation
            assert "translate [" not in content
            assert "protein_total" in content
            assert "protein_mlff" not in content  # No other components