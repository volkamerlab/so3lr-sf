"""
Pytest configuration and shared fixtures for SO3LR-SF tests.
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import numpy as np

# Note: Heavy dependencies are now mocked only when needed via fixtures

# Create a mock calculator class
class MockSo3lrSfCalculator:
    def __init__(self, *args, **kwargs):
        self.output_per_atom_energy_components = kwargs.get('output_per_atom_energy_components', False)
        self.model_path = kwargs.get('model_path', '/mock/model/path')
        self.lr_cutoff = kwargs.get('lr_cutoff', 12.0)
        self.dtype = kwargs.get('dtype', np.float32)
        self._calculator = Mock()

        # Configure mock calculator for ASE compatibility
        self._calculator.results = {}
        # Make get_potential_energy return a float instead of Mock
        self._calculator.get_potential_energy = Mock(return_value=-100.0)

    def calculate_energy(self, atoms):
        """Mock calculate_energy that always returns -100.0."""
        return -100.0

    def get_per_atom_energy_components(self):
        """Mock get_per_atom_energy_components."""
        if not self.output_per_atom_energy_components:
            raise ValueError("Per-atom energy components not enabled")

        return {
            'mlff_atomic_energy': np.array([0.1, 0.2, 0.3]),
            'zbl_repulsion': np.array([0.0, 0.0, 0.0]),
            'electrostatic_energy': np.array([0.05, 0.1, 0.15]),
            'dispersion_energy': np.array([-0.01, -0.02, -0.03])
        }

    def get_potential_energy(self):
        """Mock method for ASE compatibility."""
        return -100.0

    def _init_calculator(self):
        """Mock initialization method for structure optimization."""
        # Create a fresh mock calculator with proper return values
        self._calculator = Mock()
        self._calculator.results = {}
        self._calculator.get_potential_energy = Mock(return_value=-100.0)
        self._calculator.get_forces = Mock(return_value=np.zeros((3, 3)))

# Note: Calculator mocking is now handled by fixtures, not globally

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
    """Provide a mock So3lrSfCalculator instance for unit tests."""
    return MockSo3lrSfCalculator()


@pytest.fixture
def real_calculator(so3lr_model_path):
    """Provide real So3lrSfCalculator for integration tests."""
    from pathlib import Path
    if not Path(so3lr_model_path).exists():
        pytest.skip(f"SO3LR model not available at {so3lr_model_path}")

    from src.calculator import So3lrSfCalculator
    return So3lrSfCalculator(
        model_path=so3lr_model_path,
        output_per_atom_energy_components=False
    )


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
def water_files(test_data_dir):
    """Get water molecule files from test_data directory."""
    files = {}

    # Check if test_data directory exists
    if not test_data_dir.exists():
        pytest.skip("test_data directory not found")

    # Look for water files in different formats
    for format_ext in ['xyz', 'pdb', 'sdf']:
        water_file = test_data_dir / f"water.{format_ext}"
        if water_file.exists():
            files[format_ext] = water_file
        else:
            pytest.skip(f"Water file water.{format_ext} not found in test_data")

    return files


@pytest.fixture
def multi_water_file(test_data_dir):
    """Get multi-water file from test_data directory."""
    multi_file = test_data_dir / "multi_water.sdf"
    if not multi_file.exists():
        pytest.skip("multi_water.sdf not found in test_data")
    return multi_file


@pytest.fixture
def alanine_files(test_data_dir):
    """Get alanine molecule files from test_data directory."""
    files = {}

    # Check if test_data directory exists
    if not test_data_dir.exists():
        pytest.skip("test_data directory not found")

    # Look for alanine files in different formats
    for format_ext in ['xyz', 'pdb']:
        alanine_file = test_data_dir / f"alanine.{format_ext}"
        if alanine_file.exists():
            files[format_ext] = alanine_file

    if not files:
        pytest.skip("No alanine files found in test_data")

    return files


@pytest.fixture
def so3lr_model_path():
    """Get real SO3LR model path for testing - must exist."""
    from src.config import get_so3lr_model_path
    model_path = get_so3lr_model_path()

    if model_path is None:
        raise FileNotFoundError("SO3LR model parameters directory not found. Please install so3lr package or set SO3LR_MODEL_PATH environment variable.")

    # Verify the path actually exists
    from pathlib import Path
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"SO3LR model directory does not exist: {model_path}")

    if not path.is_dir():
        raise NotADirectoryError(f"SO3LR model path is not a directory: {model_path}")

    return model_path


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
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running test"
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