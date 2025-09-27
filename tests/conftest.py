"""
Pytest configuration and shared fixtures for SO3LR-SF tests.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock
import numpy as np

# Mock heavy dependencies BEFORE any imports
sys.modules['jax'] = Mock()
sys.modules['jax.numpy'] = Mock()
sys.modules['mlff'] = Mock()
sys.modules['mlff.md'] = Mock()
sys.modules['mlff.md.calculator_sparse'] = Mock()

# Create a mock calculator class
class MockSo3lrSfCalculator:
    def __init__(self, *args, **kwargs):
        self.output_per_atom_energy_components = kwargs.get('output_per_atom_energy_components', False)
        self.model_path = kwargs.get('model_path', '/mock/model/path')

    def calculate_energy(self, atoms):
        return -100.0

    def get_per_atom_energy_components(self):
        return {
            'mlff_atomic_energy': np.array([0.1, 0.2, 0.3]),
            'zbl_repulsion': np.array([0.0, 0.0, 0.0]),
            'electrostatic_energy': np.array([0.05, 0.1, 0.15]),
            'dispersion_energy': np.array([-0.01, -0.02, -0.03])
        }

# Mock the calculator module
sys.modules['src.calculator'] = Mock()
sys.modules['src.calculator'].So3lrSfCalculator = MockSo3lrSfCalculator

# Add src to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ase import Atoms


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_protein_atoms():
    """Create a simple protein-like atoms object for testing."""
    # Simple alanine residue
    positions = np.array([
        [0.0, 0.0, 0.0],      # N
        [1.458, 0.0, 0.0],    # CA
        [2.009, 1.421, 0.0],  # C
        [3.227, 1.468, 0.0],  # O
        [1.938, -0.507, 1.212], # CB
        [-0.364, 0.890, 0.0], # H
        [1.755, -0.507, -0.890], # HA
        [1.610, 0.003, 2.102],    # HB1
        [1.610, -1.518, 1.212],   # HB2
        [3.000, -0.507, 1.212]    # HB3
    ])

    symbols = ['N', 'C', 'C', 'O', 'C', 'H', 'H', 'H', 'H', 'H']

    return Atoms(symbols=symbols, positions=positions)


@pytest.fixture
def sample_ligand_atoms():
    """Create a simple ligand atoms object for testing (water)."""
    positions = np.array([
        [0.0, 0.0, 0.0],      # O
        [0.757, 0.586, 0.0],  # H
        [-0.757, 0.586, 0.0]  # H
    ])

    symbols = ['O', 'H', 'H']

    return Atoms(symbols=symbols, positions=positions)


@pytest.fixture
def sample_complex_atoms(sample_protein_atoms, sample_ligand_atoms):
    """Create a complex from protein + ligand."""
    # Translate ligand to avoid overlap
    ligand_translated = sample_ligand_atoms.copy()
    ligand_translated.translate([5.0, 0.0, 0.0])

    # Combine protein and ligand
    complex_atoms = sample_protein_atoms + ligand_translated

    return complex_atoms


@pytest.fixture
def mock_calculator():
    """Create a mock SO3LR calculator for testing."""
    return MockSo3lrSfCalculator(output_per_atom_energy_components=True)


@pytest.fixture
def sample_xyz_file(temp_dir, sample_protein_atoms):
    """Create a sample XYZ file for testing."""
    xyz_file = temp_dir / "sample_protein.xyz"

    from ase.io import write
    write(str(xyz_file), sample_protein_atoms)

    return xyz_file


@pytest.fixture
def sample_sdf_file(temp_dir, sample_ligand_atoms):
    """Create a sample SDF file for testing."""
    sdf_file = temp_dir / "sample_ligand.sdf"

    # Create a simple SDF content
    sdf_content = """water
  -OEChem-01012400002D

  3  2  0     0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.7570    0.5860    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7570    0.5860    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
M  END
$$$$
"""

    with open(sdf_file, 'w') as f:
        f.write(sdf_content)

    return sdf_file


@pytest.fixture
def multi_sdf_file(temp_dir):
    """Create a multi-molecule SDF file for testing."""
    multi_sdf_file = temp_dir / "multi_ligands.sdf"

    sdf_content = """water1
  -OEChem-01012400002D

  3  2  0     0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    0.7570    0.5860    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7570    0.5860    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
M  END
$$$$
water2
  -OEChem-01012400002D

  3  2  0     0  0  0  0  0  0999 V2000
    2.0000    0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
    2.7570    0.5860    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    1.2430    0.5860    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0  0  0  0
  1  3  1  0  0  0  0
M  END
$$$$
"""

    with open(multi_sdf_file, 'w') as f:
        f.write(sdf_content)

    return multi_sdf_file


@pytest.fixture
def sample_energy_components():
    """Sample per-atom energy components for testing explainability."""
    return {
        'mlff_atomic_energy': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        'zbl_repulsion': np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        'electrostatic_energy': np.array([0.05, 0.1, 0.15, 0.2, 0.25]),
        'dispersion_energy': np.array([-0.01, -0.02, -0.03, -0.04, -0.05])
    }


@pytest.fixture
def mock_logger():
    """Create a mock logger for testing."""
    logger = Mock()
    logger.info = Mock()
    logger.debug = Mock()
    logger.warning = Mock()
    logger.error = Mock()

    return logger


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables and cleanup."""
    # Set test environment
    os.environ['TESTING'] = '1'

    yield

    # Cleanup
    if 'TESTING' in os.environ:
        del os.environ['TESTING']


# Test data directory
@pytest.fixture
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / "test_data"


# Skip markers for optional dependencies
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "requires_rdkit: mark test as requiring RDKit"
    )
    config.addinivalue_line(
        "markers", "requires_model: mark test as requiring SO3LR model files"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle optional dependencies."""
    try:
        import rdkit
        rdkit_available = True
    except ImportError:
        rdkit_available = False

    # Skip RDKit tests if not available
    skip_rdkit = pytest.mark.skip(reason="RDKit not available")

    for item in items:
        if "requires_rdkit" in item.keywords and not rdkit_available:
            item.add_marker(skip_rdkit)