"""
Tests for the config module.
"""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, Mock

from src.config import get_so3lr_model_path, get_default_model_path


class TestGetSo3lrModelPath:
    """Test get_so3lr_model_path function."""

    def test_local_params_path_exists(self, temp_dir):
        """Test when local so3lr/params directory exists."""
        # Create a mock local params directory
        local_params = temp_dir / "so3lr" / "params"
        local_params.mkdir(parents=True)

        with patch('src.config.Path') as mock_path:
            # Mock Path(__file__).parent.parent to return temp_dir
            mock_path.return_value.parent.parent = temp_dir
            mock_path.side_effect = lambda x: Path(x) if isinstance(x, str) else mock_path.return_value

            # Mock the local_params_path to point to our temp directory
            mock_local_path = temp_dir / "so3lr" / "params"
            with patch.object(Path, 'exists', return_value=True), \
                 patch.object(Path, 'is_dir', return_value=True), \
                 patch.object(Path, 'resolve', return_value=mock_local_path):

                result = get_so3lr_model_path()
                assert result == str(mock_local_path)

    def test_installed_so3lr_package_primary_path(self):
        """Test when so3lr package is installed with primary path structure."""
        mock_so3lr = Mock()
        mock_so3lr.__file__ = "/path/to/so3lr/__init__.py"

        with patch.dict('sys.modules', {'so3lr': mock_so3lr}), \
             patch('src.config.Path') as mock_path_class:

            # Mock the path operations
            mock_package_path = Mock()
            mock_params_path = Mock()

            mock_path_class.side_effect = lambda x: {
                "/path/to/so3lr/__init__.py": Mock(parent=mock_package_path),
                mock_package_path / "so3lr" / "params": mock_params_path
            }.get(x, Mock())

            mock_params_path.exists.return_value = True
            mock_params_path.is_dir.return_value = True
            mock_params_path.resolve.return_value = "/resolved/path/to/params"

            result = get_so3lr_model_path()
            assert result == "/resolved/path/to/params"

    def test_installed_so3lr_package_alternative_path(self):
        """Test when so3lr package is installed with alternative path structure."""
        mock_so3lr = Mock()
        mock_so3lr.__file__ = "/path/to/so3lr/__init__.py"

        with patch.dict('sys.modules', {'so3lr': mock_so3lr}), \
             patch('src.config.Path') as mock_path_class:

            # Mock the path operations
            mock_package_path = Mock()
            mock_primary_params = Mock()
            mock_alt_params = Mock()

            def path_side_effect(x):
                if x == "/path/to/so3lr/__init__.py":
                    return Mock(parent=mock_package_path)
                elif str(x).endswith("so3lr/params"):
                    return mock_primary_params
                elif str(x).endswith("params"):
                    return mock_alt_params
                return Mock()

            mock_path_class.side_effect = path_side_effect

            # Primary path doesn't exist, alternative does
            mock_primary_params.exists.return_value = False
            mock_alt_params.exists.return_value = True
            mock_alt_params.is_dir.return_value = True
            mock_alt_params.resolve.return_value = "/resolved/alt/path/to/params"

            result = get_so3lr_model_path()
            assert result == "/resolved/alt/path/to/params"

    def test_so3lr_import_error(self):
        """Test when so3lr package is not installed."""
        with patch('src.config.Path') as mock_path:
            # Mock local path doesn't exist
            mock_path.return_value.parent.parent = Path("/fake")
            mock_local_path = Mock()
            mock_local_path.exists.return_value = False

            with patch('builtins.__import__', side_effect=ImportError), \
                 patch('os.getenv', return_value=None):

                result = get_so3lr_model_path()
                assert result is None

    def test_environment_variable_set(self):
        """Test when SO3LR_MODEL_PATH environment variable is set."""
        with patch('src.config.Path') as mock_path:
            # Mock local path doesn't exist
            mock_path.return_value.parent.parent = Path("/fake")
            mock_local_path = Mock()
            mock_local_path.exists.return_value = False

            with patch('builtins.__import__', side_effect=ImportError), \
                 patch('os.getenv', return_value="/env/path/to/models"), \
                 patch.object(Path, 'exists', return_value=True), \
                 patch.object(Path, 'resolve', return_value="/resolved/env/path"):

                result = get_so3lr_model_path()
                assert result == "/resolved/env/path"

    def test_environment_variable_invalid_path(self):
        """Test when SO3LR_MODEL_PATH points to non-existent path."""
        with patch('src.config.Path') as mock_path:
            # Mock local path doesn't exist
            mock_path.return_value.parent.parent = Path("/fake")
            mock_local_path = Mock()
            mock_local_path.exists.return_value = False

            with patch('builtins.__import__', side_effect=ImportError), \
                 patch('os.getenv', return_value="/invalid/path"), \
                 patch.object(Path, 'exists', return_value=False):

                result = get_so3lr_model_path()
                assert result is None

    def test_no_model_path_found(self):
        """Test when no model path is found anywhere."""
        with patch('src.config.Path') as mock_path:
            # Mock local path doesn't exist
            mock_path.return_value.parent.parent = Path("/fake")
            mock_local_path = Mock()
            mock_local_path.exists.return_value = False

            with patch('src.config.os.getenv', return_value=None):
                # Mock the import so3lr to fail
                with patch.dict('sys.modules', {'so3lr': None}):
                    result = get_so3lr_model_path()
                    assert result is None


class TestGetDefaultModelPath:
    """Test get_default_model_path function."""

    def test_model_path_found(self):
        """Test when model path is found."""
        with patch('src.config.get_so3lr_model_path', return_value="/path/to/model"):
            result = get_default_model_path()
            assert result == "/path/to/model"

    def test_model_path_not_found(self):
        """Test when model path is not found."""
        with patch('src.config.get_so3lr_model_path', return_value=None):
            with pytest.raises(FileNotFoundError) as exc_info:
                get_default_model_path()

            assert "SO3LR model parameters not found" in str(exc_info.value)
            assert "so3lr package is installed" in str(exc_info.value)
            assert "SO3LR_MODEL_PATH environment variable" in str(exc_info.value)