# sundials_py

Python bindings for SUNDIALS solvers using `pybind11` and `CMake`.

This project provides an interface to use ARKODE and CVODE (from the SUNDIALS suite) directly in Python, enabling integration of ODE systems with explicit and implicit solvers using familiar NumPy arrays.

## 📦 Features

- ARKODE and CVODE solvers via `pybind11`
- Support for explicit, implicit, and IMEX Butcher tables
- Integration with NumPy arrays for initial conditions and results
- Step-by-step integration and detailed solver statistics
- Easily extensible C++/Python interface

## 🛠 Installation

### 1. Clone the repo

```bash
git clone https://github.com/DMO-LAB/pysundial.git
cd sundials_py
```

### 2. Set up environment

```bash
conda create -n sundialEnv python=3.11
conda activate sundialEnv
pip install -e . --no-cache-dir
```

Ensure that SUNDIALS and its headers are available to CMake in your environment.

### 3. Build the bindings (automatically handled by pip install)

If needed manually:
```bash
mkdir -p build && cd build
cmake ..
make
```

## 🚀 Usage

Example:

```python
from sundials_py import ARKodeSolver, ButcherTable

# Define your right-hand side functions here
# Then:
solver = ARKodeSolver(system_size=3, explicit_fn=your_exp_fn, butcher_table=ButcherTable.ARK436L2SA_ERK_6_3_4)
solver.initialize(y0=numpy_array)
result = solver.solve_to(10.0)
```

See `test_implicit.py` and `test_arkode.py` for more usage examples.

## 📁 Structure

```
src/
  arkode/       # ARKODE bindings + Butcher tables
  cvode/        # CVODE bindings
  common/       # Shared utilities
  utils/        # Callback and vector wrappers
sundials_py/    # Python package entry point
tests/          # Python tests
```

## 🧪 Tests

Run:

```bash
python test_implicit.py
python test_arkode.py
```


