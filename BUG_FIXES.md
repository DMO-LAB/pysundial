# Bug Fix Log

This file documents bugs encountered in the codebase, their fixes, and relevant details.

---

## 2024-06-09
**Title:** Segmentation fault in Functional iteration (CVODE)

- **Type:** Runtime Error
- **Level:** Fatal
- **Description:**
  - Using Functional iteration (BDF/Adams + Functional) caused a segmentation fault when calling `CVode()`.
- **Fix:**
  - Explicitly created and attached a `SUNNonlinSol_FixedPoint` nonlinear solver to CVODE, as required by SUNDIALS, for Functional iteration cases.

---

## 2024-06-09
**Title:** ImportError after renaming to SundialsPy (extension module name mismatch)

- **Type:** Import Error
- **Level:** Fatal
- **Description:**
  - After renaming the project and package to `SundialsPy`, the compiled extension was built as `SundialsPy.cpython-311-darwin.so` (without a leading underscore), but the Python package expected `_SundialsPy.cpython-311-darwin.so` (with a leading underscore). This caused `ImportError: cannot import name '_SundialsPy' from partially initialized module 'SundialsPy'`.
- **Fix:**
  - Renamed the extension file to `_SundialsPy.cpython-311-darwin.so` to match the import in `__init__.py`. The build system should be updated to always produce the correct module name with a leading underscore.

---

_Add new entries below using the same format:_

## YYYY-MM-DD
**Title:** <Bug Title>

- **Type:** <Type>
- **Level:** <Fatal/Warning>
- **Description:**
  - <Describe the bug and how it manifested>
- **Fix:**
  - <Describe the fix applied> 