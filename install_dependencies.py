#!/usr/bin/env python3
"""
Automated dependency installer for PySundial.
This script tries to automatically install SUNDIALS and other system dependencies.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def run_command(cmd, shell=False):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Successfully ran: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
            return True
        else:
            print(f"✗ Failed to run: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
            print(f"  Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Exception running command: {e}")
        return False

def check_command_exists(command):
    """Check if a command exists in PATH."""
    return shutil.which(command) is not None

def detect_system():
    """Detect the operating system and available package managers."""
    system = platform.system().lower()
    
    if system == "linux":
        if Path("/etc/debian_version").exists():
            return "debian"
        elif Path("/etc/redhat-release").exists():
            return "redhat"
        elif check_command_exists("pacman"):
            return "arch"
    elif system == "darwin":
        return "macos"
    elif system == "windows":
        return "windows"
    
    return "unknown"

def install_sundials():
    """Try to install SUNDIALS using various methods."""
    print("Installing SUNDIALS...")
    
    # First try conda if available
    if check_command_exists("conda"):
        print("Found conda, attempting to install SUNDIALS via conda-forge...")
        if run_command(["conda", "install", "-c", "conda-forge", "sundials", "-y"]):
            return True
    
    # Try mamba if available
    if check_command_exists("mamba"):
        print("Found mamba, attempting to install SUNDIALS via conda-forge...")
        if run_command(["mamba", "install", "-c", "conda-forge", "sundials", "-y"]):
            return True
    
    # System-specific installations
    system = detect_system()
    
    if system == "debian":
        print("Detected Debian/Ubuntu system, installing via apt...")
        commands = [
            ["sudo", "apt-get", "update"],
            ["sudo", "apt-get", "install", "-y", "libsundials-dev"]
        ]
        for cmd in commands:
            if not run_command(cmd):
                return False
        return True
    
    elif system == "redhat":
        print("Detected RedHat/CentOS/Fedora system...")
        # Try dnf first (Fedora), then yum (CentOS/RHEL)
        if check_command_exists("dnf"):
            return run_command(["sudo", "dnf", "install", "-y", "sundials-devel"])
        elif check_command_exists("yum"):
            return run_command(["sudo", "yum", "install", "-y", "sundials-devel"])
    
    elif system == "arch":
        print("Detected Arch Linux system...")
        return run_command(["sudo", "pacman", "-S", "--noconfirm", "sundials"])
    
    elif system == "macos":
        print("Detected macOS system...")
        if check_command_exists("brew"):
            return run_command(["brew", "install", "sundials"])
        else:
            print("Homebrew not found. Please install Homebrew first:")
            print("/bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
    
    elif system == "windows":
        print("Windows detected. Please install SUNDIALS manually:")
        print("1. Install conda/miniconda")
        print("2. Run: conda install -c conda-forge sundials")
        
    return False

def install_python_dependencies():
    """Install Python dependencies."""
    print("Installing Python dependencies...")
    
    python_deps = [
        "pybind11[global]>=2.6.0",
        "numpy>=1.19.0",
        "cmake>=3.10"
    ]
    
    for dep in python_deps:
        print(f"Installing {dep}...")
        if not run_command([sys.executable, "-m", "pip", "install", dep]):
            print(f"Failed to install {dep}")
            return False
    
    return True

def verify_installation():
    """Verify that dependencies are properly installed."""
    print("Verifying installation...")
    
    # Check Python packages
    try:
        import pybind11
        import numpy
        print("✓ Python dependencies OK")
    except ImportError as e:
        print(f"✗ Python dependency missing: {e}")
        return False
    
    # Check for SUNDIALS headers (basic check)
    sundials_paths = [
        Path(os.environ.get("CONDA_PREFIX", "")) / "include" / "sundials",
        Path("/usr/local/include/sundials"),
        Path("/opt/homebrew/include/sundials"),
        Path("/usr/include/sundials")
    ]
    
    sundials_found = False
    for path in sundials_paths:
        if path.exists() and (path / "sundials_config.h").exists():
            print(f"✓ SUNDIALS headers found at: {path}")
            sundials_found = True
            break
    
    if not sundials_found:
        print("✗ SUNDIALS headers not found")
        return False
    
    return True

def main():
    print("PySundial Dependency Installer")
    print("=" * 40)
    
    # Install Python dependencies first
    if not install_python_dependencies():
        print("Failed to install Python dependencies")
        return 1
    
    # Install SUNDIALS
    if not install_sundials():
        print("Failed to install SUNDIALS")
        print("\nPlease install SUNDIALS manually and then run:")
        print("pip install -e .")
        return 1
    
    # Verify installation
    if not verify_installation():
        print("Installation verification failed")
        return 1
    
    print("\n" + "=" * 40)
    print("✓ All dependencies installed successfully!")
    print("You can now run: pip install -e .")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
