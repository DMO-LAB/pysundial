# PySundial - Python Bindings for SUNDIALS

Python bindings for the SUNDIALS suite of differential equation solvers, providing Python interfaces to CVODE, ARKODE, and other SUNDIALS solvers.

## Features

- Python bindings for SUNDIALS CVODE (ODE solver)
- Python bindings for SUNDIALS ARKODE (adaptive Runge-Kutta solver)
- NumPy integration for efficient array operations
- Modern CMake build system with automatic dependency detection

## Prerequisites

Before installing PySundial, you need to have SUNDIALS installed on your system. The installation process varies by platform:

### Installing SUNDIALS

#### Option 1: Using Conda (Recommended)
```bash
conda install -c conda-forge sundials
```

#### Option 2: Using Package Managers

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install libsundials-dev
```

**macOS with Homebrew:**
```bash
brew install sundials
```

**CentOS/RHEL/Fedora:**
```bash
# CentOS/RHEL
sudo yum install sundials-devel

# Fedora
sudo dnf install sundials-devel
```

#### Option 3: Building from Source
For the most up-to-date version, you can build SUNDIALS from source:
```bash
git clone https://github.com/LLNL/sundials.git
cd sundials
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j$(nproc)
sudo make install
```

## Installation

Once SUNDIALS is installed, you can install PySundial:

### From Source (Development)
```bash
git clone https://github.com/DMO-LAB/pysundial.git
cd pysundial
pip install -e .
```

### From PyPI (when available)
```bash
pip install sundials-py
```

## Quick Start

```python
import numpy as np
from sundials_py import cvode

# Define your ODE system
def rhs(t, y):
    return np.array([-0.5 * y[0], -0.1 * y[1]])

# Initial conditions
y0 = np.array([1.0, 0.5])
t0 = 0.0
tf = 10.0

# Solve the ODE
solver = cvode.Solver()
t, y = solver.solve(rhs, y0, t0, tf)

print(f"Solution at t={tf}: {y[-1]}")
```

## Troubleshooting

### Common Installation Issues

#### 1. SUNDIALS Not Found
If you get an error about SUNDIALS not being found:
```
SUNDIALS NOT FOUND!
```
Make sure SUNDIALS is properly installed. Try:
```bash
# For conda users
conda install -c conda-forge sundials

# For apt users (Ubuntu/Debian)
sudo apt-get install libsundials-dev
```

#### 2. pybind11 Not Found
If you get an error about pybind11:
```
PYBIND11 NOT FOUND!
```
Install pybind11 with:
```bash
pip install pybind11[global]
```

#### 3. CMake Version Issues
Make sure you have CMake 3.10 or later:
```bash
cmake --version
```
If needed, update CMake:
```bash
pip install cmake
```

### Environment-Specific Issues

#### Conda Environments
If using conda, make sure your environment is activated:
```bash
conda activate your-environment
conda install -c conda-forge sundials
pip install -e .
```

#### Virtual Environments
If using Python virtual environments:
```bash
source your-venv/bin/activate
# Install system dependencies (SUNDIALS) first
pip install pybind11[global]
pip install -e .
```

### Debugging Build Issues

To get more verbose output during installation:
```bash
pip install -e . -v
```

To see CMake configuration details:
```bash
CMAKE_VERBOSE_MAKEFILE=ON pip install -e .
```

## Development

### Setting up Development Environment
```bash
# Clone the repository
git clone https://github.com/DMO-LAB/pysundial.git
cd pysundial

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pybind11[global] numpy pytest

# Install SUNDIALS (choose one method above)
conda install -c conda-forge sundials

# Install in development mode
pip install -e .

# Run tests
pytest tests/
```

### Building Documentation
```bash
pip install sphinx sphinx-rtd-theme
cd docs
make html
```

## Requirements

- Python 3.7+
- NumPy 1.19.0+
- SUNDIALS 6.0+
- pybind11 2.6.0+
- CMake 3.10+

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for your changes
5. Run the test suite (`pytest tests/`)
6. Commit your changes (`git commit -m 'Add some amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Citation

If you use PySundial in your research, please cite:

```bibtex
@software{pysundial,
  title={PySundial: Python Bindings for SUNDIALS},
  author={Eloghosa Ikponmwoba},
  year={2023},
  url={https://github.com/DMO-LAB/pysundial}
}
```

## Acknowledgments

- [SUNDIALS](https://sundials.readthedocs.io/) - Suite of Nonlinear and Differential/Algebraic equation Solvers
- [pybind11](https://pybind11.readthedocs.io/) - Seamless operability between C++11 and Python
- [NumPy](https://numpy.org/) - Fundamental package for scientific computing with Python
