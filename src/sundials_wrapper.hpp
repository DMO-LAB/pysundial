#pragma once

// Ensure SUNDIALS types are properly defined
#include <sundials/sundials_config.h>
#include <sundials/sundials_types.h>

// Make sure at least one precision is defined
#if !defined(SUNDIALS_DOUBLE_PRECISION) && !defined(SUNDIALS_SINGLE_PRECISION) && !defined(SUNDIALS_EXTENDED_PRECISION)
#define SUNDIALS_DOUBLE_PRECISION
typedef double realtype;
#endif

// Other SUNDIALS headers
#include <nvector/nvector_serial.h>
#include <sundials/sundials_math.h>
#include <sundials/sundials_nvector.h>

// Extra sanity check - define realtype as double if it's not defined
#ifndef realtype
typedef double realtype;
#endif