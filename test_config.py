#!/usr/bin/env python3
"""
Test script for SO3LR model path configuration
"""

import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import get_so3lr_model_path, get_default_model_path

def test_config():
    print("Testing SO3LR model path configuration...")

    # Test the model path detection
    model_path = get_so3lr_model_path()
    print(f"Detected model path: {model_path}")

    if model_path:
        model_path_obj = Path(model_path)
        print(f"Path exists: {model_path_obj.exists()}")
        print(f"Is directory: {model_path_obj.is_dir()}")

        if model_path_obj.exists():
            print("Contents:")
            for item in model_path_obj.iterdir():
                print(f"  {item.name}")

    print("\nTesting default model path (with error handling):")
    try:
        default_path = get_default_model_path()
        print(f"Default model path: {default_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_config()