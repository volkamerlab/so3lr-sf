"""
Pytest configuration and fixtures for SO3LR-SF tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import numpy as np


@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def water_files(test_data_dir):
    """Dictionary of water molecule files in different formats."""
    return {
        'xyz': test_data_dir / "water.xyz",
        'pdb': test_data_dir / "water.pdb",
        'sdf': test_data_dir / "water.sdf",
        'mol2': test_data_dir / "water.mol2"
    }


@pytest.fixture
def multi_water_file(test_data_dir):
    """Multi-structure water SDF file."""
    return test_data_dir / "multi_water.sdf"


@pytest.fixture
def alanine_files(test_data_dir):
    """Dictionary of alanine amino acid files."""
    return {
        'xyz': test_data_dir / "alanine.xyz",
        'pdb': test_data_dir / "alanine.pdb"
    }


@pytest.fixture
def so3lr_model_path():
    """Universal SO3LR model path for all tests."""
    return "/home/hamza/github/so3lr-sf/so3lr/so3lr/params"


@pytest.fixture
def mock_calculator(so3lr_model_path):
    """Mock SO3LR calculator for testing without actual calculations."""
    calc = Mock()
    calc.model_path = so3lr_model_path
    calc.dtype = np.float32
    calc.output_per_atom_energy_components = False
    calc._calculator = Mock()

    # Mock energy calculation
    calc.calculate_energy.return_value = -10.5
    calc.get_per_atom_energy_components.return_value = None

    return calc


@pytest.fixture
def mock_calculator_with_components(so3lr_model_path):
    """Mock calculator with per-atom energy components."""
    calc = Mock()
    calc.model_path = so3lr_model_path
    calc.dtype = np.float32
    calc.output_per_atom_energy_components = True
    calc._calculator = Mock()

    # Mock energy calculation
    calc.calculate_energy.return_value = -10.5

    # Mock per-atom components
    components = {
        'mlff_atomic_energy': np.array([-5.0, -3.0, -2.5]),
        'zbl_repulsion': np.array([0.1, 0.05, 0.08]),
        'electrostatic_energy': np.array([-0.5, -0.3, -0.4]),
        'dispersion_energy': np.array([-0.2, -0.1, -0.15])
    }
    calc.get_per_atom_energy_components.return_value = components

    return calc


@pytest.fixture
def sample_components():
    """Sample per-atom energy components for testing."""
    return {
        'protein': {
            'mlff_atomic_energy': np.array([-15.0, -12.0, -10.0]),
            'zbl_repulsion': np.array([0.2, 0.15, 0.18]),
            'electrostatic_energy': np.array([-1.0, -0.8, -0.9]),
            'dispersion_energy': np.array([-0.3, -0.25, -0.28])
        },
        'ligand': {
            'mlff_atomic_energy': np.array([-5.0, -4.0, -3.0]),
            'zbl_repulsion': np.array([0.1, 0.08, 0.09]),
            'electrostatic_energy': np.array([-0.3, -0.25, -0.28]),
            'dispersion_energy': np.array([-0.1, -0.08, -0.09])
        },
        'complex': {
            'mlff_atomic_energy': np.array([-15.2, -12.1, -10.1, -5.1, -4.1, -3.1]),
            'zbl_repulsion': np.array([0.25, 0.18, 0.20, 0.12, 0.09, 0.10]),
            'electrostatic_energy': np.array([-1.2, -0.9, -1.0, -0.4, -0.3, -0.35]),
            'dispersion_energy': np.array([-0.4, -0.3, -0.32, -0.15, -0.12, -0.13])
        }
    }


@pytest.fixture(autouse=True)
def suppress_matplotlib():
    """Suppress matplotlib GUI during tests."""
    import matplotlib
    matplotlib.use('Agg')


@pytest.fixture
def mock_rdkit():
    """Mock RDKit functionality for testing without RDKit dependency."""
    mock_mol = Mock()
    mock_mol.GetNumAtoms.return_value = 3

    # Mock RDKit functions
    with pytest.MonkeyPatch().context() as m:
        # Mock the imports
        mock_chem = Mock()
        mock_chem.MolFromXYZFile.return_value = mock_mol
        mock_chem.MolFromMolFile.return_value = mock_mol
        mock_chem.MolFromPDBFile.return_value = mock_mol

        mock_draw = Mock()
        mock_similarity_maps = Mock()

        # Mock the drawing functions
        mock_d2d = Mock()
        mock_d2d.GetDrawingText.return_value = b"fake_image_data"
        mock_draw.MolDraw2DCairo.return_value = mock_d2d

        m.setattr("src.explainability.Chem", mock_chem)
        m.setattr("src.explainability.Draw", mock_draw)
        m.setattr("src.explainability.SimilarityMaps", mock_similarity_maps)
        m.setattr("src.explainability.rdDetermineBonds", Mock())
        m.setattr("src.explainability.rdCoordGen", Mock())

        yield {
            'chem': mock_chem,
            'draw': mock_draw,
            'similarity_maps': mock_similarity_maps,
            'mol': mock_mol
        }