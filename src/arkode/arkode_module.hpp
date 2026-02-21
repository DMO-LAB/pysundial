#pragma once

#include <pybind11/pybind11.h>
#include <arkode/arkode.h>
#include <arkode/arkode_erkstep.h>  
#include <arkode/arkode_butcher.h>
#include <arkode/arkode_arkstep.h>
#include <arkode/arkode_ls.h>

namespace py = pybind11;
namespace sundials_py {  // << Add this
// Enum for Butcher tables available in ARKODE
enum class ButcherTable {
    // Explicit methods
    HEUN_EULER_2_1_2,           // 2nd order, 2 stages
    BOGACKI_SHAMPINE_4_2_3,     // 3rd order, 4 stages
    ARK324L2SA_ERK_4_2_3,       // 3rd order, 4 stages
    ZONNEVELD_5_3_4,            // 4th order, 5 stages
    ARK436L2SA_ERK_6_3_4,       // 4th order, 6 stages
    ARK437L2SA_ERK_7_3_4,       // 4th order, 7 stages
    ARK548L2SA_ERK_8_4_5,       // 5th order, 8 stages
    VERNER_8_5_6,               // 6th order, 8 stages
    FEHLBERG_13_7_8,            // 8th order, 13 stages
    
    // Implicit methods
    BACKWARD_EULER_1_1,         // 1st order, 1 stage
    ARK2_DIRK_3_1_2, 
    SDIRK_2_1_2,                // 2nd order, 2 stages
    IMPLICIT_MIDPOINT_1_2,    // 2nd order, 2 stages
    IMPLICIT_TRAPEZOIDAL_2_2,    // 2nd order, 2 stages
    BILLINGTON_3_3_2,           // 2nd order, 3 stages
    TRBDF2_3_3_2,               // 2nd order, 3 stages
    ESDIRK325L2SA_5_2_3,        // 2nd order, 5 stages
    ESDIRK324L2SA_4_2_3,        // 2nd order, 4 stages
    ESDIRK32I5L2SA_5_2_3,       // 2nd order, 5 stages
    KVAERNO_4_2_3,              // 3rd order, 4 stages
    ARK324L2SA_DIRK_4_2_3,      // 3rd order, 4 stages
    ESDIRK436L2SA_6_3_4,        // 3rd order, 6 stages
    CASH_5_2_4,                 // 4th order, 5 stages
    CASH_5_3_4,                 // 4th order, 5 stages
    SDIRK_5_3_4,                // 4th order, 5 stages
    KVAERNO_5_3_4,              // 4th order, 5 stages
    ARK436L2SA_DIRK_6_3_4,      // 4th order, 6 stages
    ARK437L2SA_DIRK_7_3_4,      // 4th order, 7 stages
    ESDIRK43I6L2SA_6_3_4,       // 4th order, 6 stages
    QESDIRK436L2SA_6_3_4,       // 4th order, 6 stages
    ESDIRK437L2SA_7_3_4,        // 4th order, 7 stages
    ESDIRK547L2SA2_7_4_5,       // 5th order, 7 stages
    KVAERNO_7_4_5,              // 5th order, 7 stages
    ARK548L2SA_DIRK_8_4_5,      // 5th order, 8 stages
    
    // ImEx methods
    ARK324L2SA_ERK_4_2_3_DIRK_4_2_3,    // 3rd order, 4 stages
    ARK436L2SA_ERK_6_3_4_DIRK_6_3_4,    // 4th order, 6 stages
    ARK437L2SA_ERK_7_3_4_DIRK_7_3_4,    // 4th order, 7 stages
    ARK548L2SA_ERK_8_4_5_DIRK_8_4_5     // 5th order, 8 stages
};

// Butcher table utility function declarations
ARKODE_ERKTableID get_erk_table_id(ButcherTable table);
ARKODE_DIRKTableID get_dirk_table_id(ButcherTable table);
bool is_imex_pair(ButcherTable table);
bool is_explicit_method(ButcherTable table);
bool is_implicit_method(ButcherTable table);
std::string get_butcher_table_description(ButcherTable table);

// Forward declaration of the module initialization function
void init_arkode_module(py::module_ &m);

} // namespace sundials_py