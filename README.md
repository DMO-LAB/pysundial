# SundialsPy

SundialsPy is a Python interface to the SUNDIALS suite of ODE and DAE solvers, providing high-performance, flexible, and user-friendly access to advanced time integration algorithms for scientific computing and engineering applications.

## Features
- Pythonic interface to SUNDIALS core solvers
- Support for both stiff and non-stiff ODEs
- Access to CVODE and ARKODE integrators
- Flexible right-hand side (RHS) and Jacobian callback support
- Vector and array tolerances
- Easy installation and usage

## Installation

### Prerequisites
- Python 3.8+
- [SUNDIALS library](https://computing.llnl.gov/projects/sundials/sundials-software) (should be installed and available on your system)
- C++ compiler (for building the extension)
- [pybind11](https://github.com/pybind/pybind11) (usually installed automatically)
- numpy

### Install via pip (editable/development mode)
```bash
pip install -e .
```

Or, for a standard install:
```bash
pip install .
```

If you need to specify SUNDIALS include/library paths, set the following environment variables before installing:
```bash
export SUNDIALS_ROOT=/path/to/sundials
export SUNDIALS_INCLUDE_DIR=$SUNDIALS_ROOT/include
export SUNDIALS_LIBRARY_DIR=$SUNDIALS_ROOT/lib
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


