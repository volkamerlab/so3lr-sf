"""
Configuration module for SO3LR-SF

This module provides automatic detection of SO3LR model parameters from the installed package.
"""

import os
from pathlib import Path
from typing import Optional


def get_so3lr_model_path() -> Optional[str]:
    """
    Get SO3LR model parameters path from local project or installed package.

    This function finds the model path by checking:
    1. Local so3lr-sf/so3lr/params directory (downloaded by setup.py)
    2. Installed so3lr package
    3. SO3LR_MODEL_PATH environment variable

    Returns:
        str: Path to SO3LR parameters directory, or None if not found

    Example:
        >>> model_path = get_so3lr_model_path()
        >>> print(f"Model path: {model_path}")
        Model path: /path/to/so3lr-sf/so3lr/params
    """
    # First priority: Check local project params directory
    current_dir = Path(__file__).parent.parent  # Go up from src/ to project root
    local_params_path = current_dir / "so3lr" / "params"

    if local_params_path.exists() and local_params_path.is_dir():
        return str(local_params_path.resolve())

    # Second priority: Try installed so3lr package
    try:
        import so3lr

        # Get the package installation path
        so3lr_package_path = Path(so3lr.__file__).parent

        # Construct path to model parameters
        params_path = so3lr_package_path / "so3lr" / "params"

        if params_path.exists() and params_path.is_dir():
            return str(params_path.resolve())
        else:
            # Alternative path structure
            params_path = so3lr_package_path / "params"
            if params_path.exists() and params_path.is_dir():
                return str(params_path.resolve())

    except ImportError:
        pass

    # Third priority: Check environment variable
    env_path = os.getenv('SO3LR_MODEL_PATH')
    if env_path and Path(env_path).exists():
        return str(Path(env_path).resolve())

    return None


def get_default_model_path() -> str:
    """
    Get the default model path with error handling.

    Returns:
        str: Path to model parameters

    Raises:
        FileNotFoundError: If model parameters cannot be found
    """
    model_path = get_so3lr_model_path()

    if model_path is None:
        raise FileNotFoundError(
            "SO3LR model parameters not found. "
            "Please ensure so3lr package is installed or set SO3LR_MODEL_PATH environment variable."
        )

    return model_path


# Default configuration
DEFAULT_SO3LR_MODEL_PATH = get_so3lr_model_path()