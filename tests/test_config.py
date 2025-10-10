"""
Tests for the config module.
"""

import sys
import pytest
from pathlib import Path
from unittest.mock import patch, Mock

from src.config import get_so3lr_model_path, get_default_model_path


class TestGetSo3lrModelPath:
    """Test get_so3lr_model_path function using real filesystem operations."""

    def test_local_params_path_exists(self, temp_dir):
        """Test when local so3lr/params directory exists."""
        # Create actual local params directory
        local_params = temp_dir / "so3lr" / "params"
        local_params.mkdir(parents=True)

        # Create a dummy file to make it a valid directory
        (local_params / "dummy.txt").write_text("test")

        # Patch the config module to use our temp directory as project root
        with patch('src.config.Path') as mock_path:
            def path_side_effect(x):
                if str(x).endswith('config.py'):
                    # Return a mock that makes parent.parent point to our temp_dir
                    mock_file_path = Mock()
                    mock_file_path.parent.parent = temp_dir
                    return mock_file_path
                else:
                    return Path(x)

            mock_path.side_effect = path_side_effect

            result = get_so3lr_model_path()
            assert result == str((temp_dir / "so3lr" / "params").resolve())

    def test_installed_so3lr_package_primary_path(self, temp_dir):
        """Test when so3lr package is installed with primary path structure."""
        # Create so3lr package structure
        so3lr_package_dir = temp_dir / "so3lr_package"
        params_dir = so3lr_package_dir / "so3lr" / "params"
        params_dir.mkdir(parents=True)
        (params_dir / "dummy.txt").write_text("test")

        # Mock so3lr module
        mock_so3lr = Mock()
        mock_so3lr.__file__ = str(so3lr_package_dir / "__init__.py")

        # Mock local path to not exist
        fake_project_root = temp_dir / "fake_project"
        fake_project_root.mkdir()

        with patch.dict('sys.modules', {'so3lr': mock_so3lr}), \
             patch('src.config.Path') as mock_path:

            def path_side_effect(x):
                if str(x).endswith('config.py'):
                    mock_file_path = Mock()
                    mock_file_path.parent.parent = fake_project_root
                    return mock_file_path
                else:
                    return Path(x)

            mock_path.side_effect = path_side_effect

            result = get_so3lr_model_path()
            assert result == str(params_dir.resolve())

    def test_installed_so3lr_package_alternative_path(self, temp_dir):
        """Test when so3lr package has alternative path structure."""
        # Create alternative so3lr package structure
        so3lr_package_dir = temp_dir / "so3lr_package"
        params_dir = so3lr_package_dir / "params"  # Direct params, not so3lr/params
        params_dir.mkdir(parents=True)
        (params_dir / "dummy.txt").write_text("test")

        mock_so3lr = Mock()
        mock_so3lr.__file__ = str(so3lr_package_dir / "__init__.py")

        fake_project_root = temp_dir / "fake_project"
        fake_project_root.mkdir()

        with patch.dict('sys.modules', {'so3lr': mock_so3lr}), \
             patch('src.config.Path') as mock_path:

            def path_side_effect(x):
                if str(x).endswith('config.py'):
                    mock_file_path = Mock()
                    mock_file_path.parent.parent = fake_project_root
                    return mock_file_path
                else:
                    return Path(x)

            mock_path.side_effect = path_side_effect

            result = get_so3lr_model_path()
            assert result == str(params_dir.resolve())

    def test_environment_variable_set(self, temp_dir):
        """Test when SO3LR_MODEL_PATH environment variable is set."""
        # Create env params directory
        env_params = temp_dir / "env_params"
        env_params.mkdir()
        (env_params / "dummy.txt").write_text("test")

        fake_project_root = temp_dir / "fake_project"
        fake_project_root.mkdir()

        with patch('src.config.Path') as mock_path, \
             patch('src.config.os.getenv', return_value=str(env_params)), \
             patch.dict('sys.modules', {'so3lr': None}):

            def path_side_effect(x):
                if str(x).endswith('config.py'):
                    mock_file_path = Mock()
                    mock_file_path.parent.parent = fake_project_root
                    return mock_file_path
                else:
                    return Path(x)

            mock_path.side_effect = path_side_effect

            result = get_so3lr_model_path()
            assert result == str(env_params.resolve())


    def test_environment_variable_invalid_path(self, temp_dir):
        """Test when SO3LR_MODEL_PATH points to non-existent path."""
        fake_project_root = temp_dir / "fake_project"
        fake_project_root.mkdir()

        with patch('src.config.Path') as mock_path, \
             patch('src.config.os.getenv', return_value="/nonexistent/path"), \
             patch.dict('sys.modules', {'so3lr': None}):

            def path_side_effect(x):
                if str(x).endswith('config.py'):
                    mock_file_path = Mock()
                    mock_file_path.parent.parent = fake_project_root
                    return mock_file_path
                else:
                    return Path(x)

            mock_path.side_effect = path_side_effect

            result = get_so3lr_model_path()
            assert result is None

    def test_no_model_path_found(self, temp_dir):
        """Test when no model path is found anywhere."""
        fake_project_root = temp_dir / "fake_project"
        fake_project_root.mkdir()

        with patch('src.config.Path') as mock_path, \
             patch('src.config.os.getenv', return_value=None), \
             patch.dict('sys.modules', {'so3lr': None}):

            def path_side_effect(x):
                if str(x).endswith('config.py'):
                    mock_file_path = Mock()
                    mock_file_path.parent.parent = fake_project_root
                    return mock_file_path
                else:
                    return Path(x)

            mock_path.side_effect = path_side_effect

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