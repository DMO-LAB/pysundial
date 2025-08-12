# SundialsPy – Development Notes

## Overview

SundialsPy is a pybind11-based Python interface to the SUNDIALS suite. The package builds a C++ extension `_SundialsPy` and exposes it via `SundialsPy` with `cvode` and `arkode` submodules.

- C++ sources: `src/`
  - `src/module.cpp`: pybind11 module, submodules, SUNContext init/cleanup
  - `src/common/*`: SUNContext, NumPy/N_Vector conversion, error helpers
  - `src/utils/callback_wrappers.hpp`: Python callback wrappers (RHS, Jacobian, roots)
  - `src/cvode/*`: CVODE solver class and bindings
  - `src/arkode/*`: ARKode solver class; Butcher table enums/mapping
- Python: `SundialsPy/__init__.py` lifts extension symbols
- Build: `setup.py` + `CMakeLists.txt` (Conda/Homebrew/system SUNDIALS paths)
- Tests: `tests/` cover CVODE/ARKode explicit and implicit flows

## Public API (Python)

- `SundialsPy.cvode.CVodeSolver(system_size, rhs_fn, iter_type, linsol_type, use_bdf)`
  - Methods: `initialize`, `solve_to`, `solve`, `set_jacobian`, `set_root_function`, `get_stats`, `get_last_step`, `get_current_time`
- `SundialsPy.arkode.ARKodeSolver(system_size, explicit_fn, implicit_fn=None, butcher_table, linsol_type)`
  - Methods: `initialize`, `solve_to`, `solve_sequence`, `integrate_to_time`, `advance_one_step`, `set_jacobian`, `set_root_function`, `set_fixed_step_size`, `set_max_num_steps`, `get_stats`, `get_butcher_info`, `get_last_step`

## Callback and Context Model

- A single global `SUNContext` is created on module import and freed at exit.
- C++ wrappers acquire the GIL, convert arrays, and validate sizes.
- User callbacks (RHS/Jacobian/roots) are stored in user_data structs passed into SUNDIALS.

## Recent Changes

- Tests added:
  - `tests/test_cvode_jacobian.py`: verifies CVODE with user Jacobian on a 2×2 linear system
  - `tests/test_cvode_roots.py`: verifies CVODE root function doesn’t break integration
  - `tests/test_arkode_roots.py`: verifies ARKode root function integrates safely
- CVODE: unified user_data so RHS/Jacobian/Roots share one struct; added CVODE-specific wrappers. Prevents user_data clobbering.
  - Files: `src/utils/callback_wrappers.hpp`, `src/cvode/cvode_module.cpp`
- ARKode: root callbacks now stored in main `PyArkFnData` and registered via `ARKodeRootInit`, avoiding mixed user_data.
  - Files: `src/utils/callback_wrappers.hpp`, `src/arkode/arkode_module.cpp`
- Test status: all tests pass (15/15).

## Potential Improvements

- Apply `ButcherTable` to the ARKode stepper (explicit/implicit/imex table selection is mapped but not set on the solver instance).
- Implement additional linear solvers (band/sparse/iterative) or reduce the public enum to implemented options.
- Build portability: consider letting CMake own Python detection fully and avoid hardcoding `PYTHON_LIBRARY`; relax `NO_DEFAULT_PATH` for SUNDIALS if appropriate.
- Reduce or gate debug prints in wrappers.

## Dev Tips

- Build (editable): `pip install -e .`
- Run tests: `pytest tests/`
- Clean build artifacts: remove `build/`, `dist/`, `SundialsPy.egg-info/` before rebuilds if needed.
