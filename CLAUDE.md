# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

### Build from Source
```bash
# Install dependencies first (requires SUNDIALS, CMake, pybind11)
conda install -c conda-forge sundials pybind11 cmake numpy  # Conda (recommended)
# OR
brew install sundials cmake && pip install pybind11[global] numpy  # macOS
# OR  
sudo apt-get install libsundials-dev cmake build-essential && pip install pybind11[global] numpy  # Ubuntu

# Build and install
pip install -e .  # Development install
# OR
pip install .     # Regular install
```

### Testing
```bash
pytest tests/                    # Run all tests
pytest tests/test_cvode.py      # Run CVODE tests only
pytest tests/test_arkode.py     # Run ARKode tests only
```

### Package Building
```bash
python -m build                 # Build source and wheel distributions
```

## Project Architecture

### Core Structure
- **SundialsPy/**: Python package directory containing the main `__init__.py` that imports the C++ extension `_SundialsPy`
- **src/**: C++ source code using pybind11 bindings
  - **module.cpp**: Main pybind11 module definition, creates `cvode` and `arkode` submodules
  - **cvode/**: CVODE solver implementation (BDF/Adams methods, Newton/Functional iteration) 
  - **arkode/**: ARKode solver implementation (explicit/implicit/IMEX Runge-Kutta methods)
  - **common/**: Shared utilities and SUNDIALS context management
  - **utils/**: Callback and vector wrapper utilities

### Build System
- Uses **CMake** with custom `setup.py` that invokes CMake during `pip install`
- Platform-specific SUNDIALS library detection (Conda, Homebrew, system paths)
- Builds C++ extension named `_SundialsPy` (note the underscore prefix)

### Key Components

#### CVODE Module (`src/cvode/`)
- Supports BDF and Adams multistep methods
- Newton and Functional iteration types  
- Dense, Band, Sparse, Iterative linear solvers (SPGMR, SPBCG, SPTFQMR, PCG)
- Automatic nonlinear solver setup for Functional iteration

#### ARKode Module (`src/arkode/`)
- Explicit, Implicit, and IMEX Runge-Kutta methods
- Extensive Butcher table support (see `arkode_module.hpp` for available tables)
- Adaptive time stepping

#### Python Interface
- Pythonic callback system for RHS functions and Jacobians
- NumPy array integration for state vectors and tolerances
- Exception handling with proper SUNDIALS context cleanup

### Dependencies
- **SUNDIALS** (v6.6.2+): Core numerical solvers
- **pybind11**: C++/Python bindings  
- **CMake** (3.10+): Build system
- **NumPy**: Array operations

### Testing Strategy
- Parametrized tests covering different solver configurations
- Comparison against analytical solutions for validation
- Tests for both CVODE and ARKode integrators with various methods

## Platform Support
- Linux (x86_64)
- macOS (x86_64, arm64) 
- Windows (x86_64)
- Python 3.8-3.12

## Installation Troubleshooting
Common issues documented in README.md:
- SUNDIALS library detection problems
- pybind11 configuration issues  
- Platform-specific compiler/linker problems