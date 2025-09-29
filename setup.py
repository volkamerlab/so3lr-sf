#!/usr/bin/env python3
"""
Setup script for so3lr-sf
Installs dependencies with Poetry and downloads SO3LR model parameters
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up so3lr-sf environment...")

    # Step 1: Install dependencies with Poetry
    if not run_command("poetry install --with test", "Installing dependencies with Poetry"):
        sys.exit(1)

    # Step 2: Download SO3LR model parameters
    download_cmd = 'mkdir -p so3lr && cd so3lr && curl -L https://github.com/general-molecular-simulations/so3lr/archive/main.tar.gz | tar -xz --strip-components=2 so3lr-main/so3lr/params'
    if not run_command(download_cmd, "Downloading SO3LR model parameters"):
        sys.exit(1)

    # Step 3: Verify setup
    if Path("so3lr/params").exists():
        print("✅ Setup completed successfully!")
        print("📁 SO3LR model parameters are available in: so3lr/params/")
        print("🧪 You can now run tests with: poetry run pytest")
    else:
        print("❌ Setup completed but model parameters not found")
        sys.exit(1)

if __name__ == "__main__":
    main()