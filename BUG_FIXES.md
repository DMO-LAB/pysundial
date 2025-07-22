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

_Add new entries below using the same format:_

## YYYY-MM-DD
**Title:** <Bug Title>

- **Type:** <Type>
- **Level:** <Fatal/Warning>
- **Description:**
  - <Describe the bug and how it manifested>
- **Fix:**
  - <Describe the fix applied> 