# SundialsPy

SundialsPy is a Python interface to the SUNDIALS suite of ODE and DAE solvers, providing high-performance, flexible, and user-friendly access to advanced time integration algorithms for scientific computing and engineering applications.

## Features
- Pythonic interface to SUNDIALS core solvers
- Support for both stiff and non-stiff ODEs
- Access to CVODE and ARKODE integrators
- Flexible right-hand side (RHS) and Jacobian callback support
- Vector and array tolerances
- Easy installation and usage

# SundialsPy Installation Guide

## Quick Installation (Recommended)

If binary wheels are available for your platform:

```bash
pip install SundialsPy
```

## Installation from Source

If you need to build from source or binary wheels aren't available:

### Prerequisites

#### Option 1: Using Conda (Recommended)
```bash
conda install -c conda-forge sundials pybind11 cmake numpy
pip install SundialsPy
```

#### Option 2: Using System Package Managers

**macOS (Homebrew):**
```bash
brew install sundials cmake
pip install pybind11[global] numpy
pip install SundialsPy
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install libsundials-dev cmake build-essential
pip install pybind11[global] numpy
pip install SundialsPy
```

**Windows:**
1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Install [CMake](https://cmake.org/download/)
3. Download and install SUNDIALS from [here](https://computing.llnl.gov/projects/sundials/sundials-software)
4. Set environment variables:
   ```cmd
   set SUNDIALS_ROOT=C:\path\to\sundials
   set SUNDIALS_INCLUDE_DIR=C:\path\to\sundials\include
   set SUNDIALS_LIBRARY_DIR=C:\path\to\sundials\lib
   ```
5. Install SundialsPy:
   ```cmd
   pip install pybind11[global] numpy
   pip install SundialsPy
   ```

### Manual SUNDIALS Installation

If SUNDIALS is not available through package managers:

```bash
# Download SUNDIALS
wget https://github.com/LLNL/sundials/releases/download/v6.6.2/sundials-6.6.2.tar.gz
tar -xzf sundials-6.6.2.tar.gz
cd sundials-6.6.2

# Build and install
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_SHARED_LIBS=ON
make -j$(nproc)  # Linux/macOS
# make -j%NUMBER_OF_PROCESSORS%  # Windows
sudo make install  # Linux/macOS (or run as admin on Windows)
```

Then install SundialsPy:
```bash
pip install pybind11[global] numpy
pip install SundialsPy
```

## Troubleshooting

### Common Issues

1. **"SUNDIALS not found" error:**
   - Make sure SUNDIALS is installed and in your PATH
   - Set environment variables manually:
     ```bash
     export SUNDIALS_ROOT=/path/to/sundials
     export SUNDIALS_INCLUDE_DIR=/path/to/sundials/include
     export SUNDIALS_LIBRARY_DIR=/path/to/sundials/lib
     ```

2. **"pybind11 not found" error:**
   ```bash
   pip install "pybind11[global]"
   ```

3. **Import errors:**
   - Make sure you're using the correct Python environment
   - Try reinstalling: `pip uninstall SundialsPy && pip install SundialsPy`

### Supported Platforms

- ✅ Linux (x86_64)
- ✅ macOS (x86_64, arm64)
- ✅ Windows (x86_64)
- ✅ Python 3.8-3.12

## Development Installation

For developers who want to modify the code:

```bash
git clone https://github.com/yourusername/SundialsPy.git
cd SundialsPy

# Install dependencies
conda install -c conda-forge sundials pybind11 cmake numpy

# Install in development mode
pip install -e .
```

## Docker Installation

For a completely isolated environment:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libsundials-dev \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install SundialsPy

# Test the installation
RUN python -c "import SundialsPy; print('SundialsPy installed successfully!')"
```

## Usage Example

```python
import numpy as np
import SundialsPy

# Define a simple ODE: dy/dt = -0.5*y

def rhs(t, y):
    return np.array([-0.5 * y[0]])

# Initial condition
y0 = np.array([1.0])
t0 = 0.0
t_end = 2.0

# Create a CVODE solver (BDF + Newton)
solver = SundialsPy.cvode.CVodeSolver(
    system_size=1,
    rhs_fn=rhs,
    iter_type=SundialsPy.cvode.IterationType.NEWTON,
    linsol_type=SundialsPy.cvode.LinearSolverType.DENSE,
    use_bdf=True
)

solver.initialize(y0, t0, 1e-6, np.array([1e-8]))
result = solver.solve_to(0.1)
print("Solution at t=0.1:", result)
```

## Available Integrators

### CVODE
- **BDF (Backward Differentiation Formula):** For stiff ODEs
- **Adams:** For non-stiff ODEs
- **Iteration Types:** Newton, Functional
- **Linear Solvers:** Dense, Band, Sparse, Iterative (SPGMR, SPBCG, SPTFQMR, PCG)

### ARKODE
- **Explicit, Implicit, and IMEX Runge-Kutta methods**
- **Butcher tables for various schemes**

## Directory Structure
- `SundialsPy/` — Python package
- `src/` — C++ source code and pybind11 bindings
- `examples/` — Example scripts

## Contributing
Contributions are welcome! Please open issues or pull requests for bug reports, feature requests, or improvements.

## License
[MIT License](LICENSE)

## Acknowledgments
- [SUNDIALS](https://computing.llnl.gov/projects/sundials) by LLNL
- [pybind11](https://github.com/pybind/pybind11)

## Acknowledgments

- [SUNDIALS](https://sundials.readthedocs.io/) - Suite of Nonlinear and Differential/Algebraic equation Solvers
- [pybind11](https://pybind11.readthedocs.io/) - Seamless operability between C++11 and Python
- [NumPy](https://numpy.org/) - Fundamental package for scientific computing with Python
