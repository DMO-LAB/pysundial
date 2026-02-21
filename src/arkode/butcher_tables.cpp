#include "arkode_module.hpp"
#include <arkode/arkode_butcher_dirk.h>
#include <arkode/arkode_butcher_erk.h>
#include <arkode/arkode_butcher.h>
#include <iostream>

namespace sundials_py {

// Function to map our enum to SUNDIALS ERK table ID
ARKODE_ERKTableID get_erk_table_id(ButcherTable table) {
    switch (table) {
        // Explicit methods
        case ButcherTable::HEUN_EULER_2_1_2:
            return ARKODE_HEUN_EULER_2_1_2;
        case ButcherTable::BOGACKI_SHAMPINE_4_2_3:
            return ARKODE_BOGACKI_SHAMPINE_4_2_3;
        case ButcherTable::ARK324L2SA_ERK_4_2_3:
            return ARKODE_ARK324L2SA_ERK_4_2_3;
        case ButcherTable::ZONNEVELD_5_3_4:
            return ARKODE_ZONNEVELD_5_3_4;
        case ButcherTable::ARK436L2SA_ERK_6_3_4:
            return ARKODE_ARK436L2SA_ERK_6_3_4;
        case ButcherTable::ARK437L2SA_ERK_7_3_4:
            return ARKODE_ARK437L2SA_ERK_7_3_4;
        case ButcherTable::ARK548L2SA_ERK_8_4_5:
            return ARKODE_ARK548L2SA_ERK_8_4_5;
        case ButcherTable::VERNER_8_5_6:
            return ARKODE_VERNER_8_5_6;
        case ButcherTable::FEHLBERG_13_7_8:
            return ARKODE_FEHLBERG_13_7_8;
            
        // ImEx pairs - return the ERK part
        case ButcherTable::ARK324L2SA_ERK_4_2_3_DIRK_4_2_3:
            return ARKODE_ARK324L2SA_ERK_4_2_3;
        case ButcherTable::ARK436L2SA_ERK_6_3_4_DIRK_6_3_4:
            return ARKODE_ARK436L2SA_ERK_6_3_4;
        case ButcherTable::ARK437L2SA_ERK_7_3_4_DIRK_7_3_4:
            return ARKODE_ARK437L2SA_ERK_7_3_4;
        case ButcherTable::ARK548L2SA_ERK_8_4_5_DIRK_8_4_5:
            return ARKODE_ARK548L2SA_ERK_8_4_5;
            
        default:
            // Default to a common 4th order method
            return ARKODE_ARK436L2SA_ERK_6_3_4;
    }
}

// Function to map our enum to SUNDIALS DIRK table ID
ARKODE_DIRKTableID get_dirk_table_id(ButcherTable table) {
    switch (table) {
        // Implicit methods
        case ButcherTable::BACKWARD_EULER_1_1:
            return ARKODE_BACKWARD_EULER_1_1;
        case ButcherTable::ARK2_DIRK_3_1_2:
            return ARKODE_ARK2_DIRK_3_1_2;
        case ButcherTable::SDIRK_2_1_2:
            return ARKODE_SDIRK_2_1_2;
        case ButcherTable::IMPLICIT_MIDPOINT_1_2:
            return ARKODE_IMPLICIT_MIDPOINT_1_2;
        case ButcherTable::IMPLICIT_TRAPEZOIDAL_2_2:
            return ARKODE_IMPLICIT_TRAPEZOIDAL_2_2;
        case ButcherTable::ESDIRK325L2SA_5_2_3:
            return ARKODE_ESDIRK325L2SA_5_2_3;
        case ButcherTable::ESDIRK324L2SA_4_2_3:
            return ARKODE_ESDIRK324L2SA_4_2_3;
        case ButcherTable::ESDIRK32I5L2SA_5_2_3:
            return ARKODE_ESDIRK32I5L2SA_5_2_3;
        case ButcherTable::BILLINGTON_3_3_2:
            return ARKODE_BILLINGTON_3_3_2;
        case ButcherTable::TRBDF2_3_3_2:
            return ARKODE_TRBDF2_3_3_2;
        case ButcherTable::KVAERNO_4_2_3:
            return ARKODE_KVAERNO_4_2_3;
        case ButcherTable::ARK324L2SA_DIRK_4_2_3:
            return ARKODE_ARK324L2SA_DIRK_4_2_3;
        case ButcherTable::ESDIRK436L2SA_6_3_4:
            return ARKODE_ESDIRK436L2SA_6_3_4;
        case ButcherTable::ESDIRK43I6L2SA_6_3_4:
            return ARKODE_ESDIRK43I6L2SA_6_3_4;
        case ButcherTable::QESDIRK436L2SA_6_3_4:
            return ARKODE_QESDIRK436L2SA_6_3_4;
        case ButcherTable::CASH_5_2_4:
            return ARKODE_CASH_5_2_4;
        case ButcherTable::CASH_5_3_4:
            return ARKODE_CASH_5_3_4;
        case ButcherTable::SDIRK_5_3_4:
            return ARKODE_SDIRK_5_3_4;
        case ButcherTable::KVAERNO_5_3_4:
            return ARKODE_KVAERNO_5_3_4;
        case ButcherTable::ARK436L2SA_DIRK_6_3_4:
            return ARKODE_ARK436L2SA_DIRK_6_3_4;
        case ButcherTable::ESDIRK437L2SA_7_3_4:
            return ARKODE_ESDIRK437L2SA_7_3_4;
        case ButcherTable::ARK437L2SA_DIRK_7_3_4:
            return ARKODE_ARK437L2SA_DIRK_7_3_4;
        case ButcherTable::ESDIRK547L2SA2_7_4_5:
            return ARKODE_ESDIRK547L2SA2_7_4_5;
        case ButcherTable::KVAERNO_7_4_5:
            return ARKODE_KVAERNO_7_4_5;
        case ButcherTable::ARK548L2SA_DIRK_8_4_5:
            return ARKODE_ARK548L2SA_DIRK_8_4_5;
            
        // ImEx pairs - return the DIRK part
        case ButcherTable::ARK324L2SA_ERK_4_2_3_DIRK_4_2_3:
            return ARKODE_ARK324L2SA_DIRK_4_2_3;
        case ButcherTable::ARK436L2SA_ERK_6_3_4_DIRK_6_3_4:
            return ARKODE_ARK436L2SA_DIRK_6_3_4;
        case ButcherTable::ARK437L2SA_ERK_7_3_4_DIRK_7_3_4:
            return ARKODE_ARK437L2SA_DIRK_7_3_4;
        case ButcherTable::ARK548L2SA_ERK_8_4_5_DIRK_8_4_5:
            return ARKODE_ARK548L2SA_DIRK_8_4_5;
            
        default:
            // Default to a common 4th order method
            return ARKODE_ARK436L2SA_DIRK_6_3_4;
    }
}

// Function to check if a Butcher table is an ImEx pair
bool is_imex_pair(ButcherTable table) {
    switch (table) {
        case ButcherTable::ARK324L2SA_ERK_4_2_3_DIRK_4_2_3:
        case ButcherTable::ARK436L2SA_ERK_6_3_4_DIRK_6_3_4:
        case ButcherTable::ARK437L2SA_ERK_7_3_4_DIRK_7_3_4:
        case ButcherTable::ARK548L2SA_ERK_8_4_5_DIRK_8_4_5:
            return true;
        default:
            return false;
    }
}

// Get Butcher table description (order, stages)
std::string get_butcher_table_description(ButcherTable table) {
    switch (table) {
        case ButcherTable::HEUN_EULER_2_1_2:
            return "Heun-Euler 2-1-2: 2nd order, 2 stages (Explicit)";
        case ButcherTable::BOGACKI_SHAMPINE_4_2_3:
            return "Bogacki-Shampine 4-2-3: 3rd order, 4 stages (Explicit)";
        case ButcherTable::ARK324L2SA_ERK_4_2_3:
            return "ARK3(2)4L[2]SA-ERK: 3rd order, 4 stages (Explicit)";
        case ButcherTable::ZONNEVELD_5_3_4:
            return "Zonneveld 5-3-4: 4th order, 5 stages (Explicit)";
        case ButcherTable::ARK436L2SA_ERK_6_3_4:
            return "ARK4(3)6L[2]SA-ERK: 4th order, 6 stages (Explicit)";
        case ButcherTable::ARK437L2SA_ERK_7_3_4:
            return "ARK4(3)7L[2]SA-ERK: 4th order, 7 stages (Explicit)";
        case ButcherTable::ARK548L2SA_ERK_8_4_5:
            return "ARK5(4)8L[2]SA-ERK: 5th order, 8 stages (Explicit)";
        case ButcherTable::VERNER_8_5_6:
            return "Verner 8-5-6: 6th order, 8 stages (Explicit)";
        case ButcherTable::FEHLBERG_13_7_8:
            return "Fehlberg 13-7-8: 8th order, 13 stages (Explicit)";

        case ButcherTable::BACKWARD_EULER_1_1:
            return "Backward Euler 1-1: 1st order, 1 stage (Implicit)";
        case ButcherTable::ARK2_DIRK_3_1_2:
            return "ARK2-DIRK 3-1-2: 2nd order, 3 stages (Implicit)";
        case ButcherTable::SDIRK_2_1_2:
            return "SDIRK-2-1-2: 2nd order, 2 stages (Implicit)";
        case ButcherTable::IMPLICIT_MIDPOINT_1_2:
            return "Implicit Midpoint 1-2: 2nd order, 2 stages (Implicit)";
        case ButcherTable::IMPLICIT_TRAPEZOIDAL_2_2:
            return "Implicit Trapezoidal 2-2: 2nd order, 2 stages (Implicit)";
        case ButcherTable::ESDIRK325L2SA_5_2_3:
            return "ESDIRK3(2)5L[2]SA-5-2-3: 2nd order, 5 stages (Implicit)";
        case ButcherTable::ESDIRK324L2SA_4_2_3:
            return "ESDIRK3(2)4L[2]SA-4-2-3: 2nd order, 4 stages (Implicit)";
        case ButcherTable::ESDIRK32I5L2SA_5_2_3:
            return "ESDIRK3(2)I5L[2]SA-5-2-3: 2nd order, 5 stages (Implicit)";
        case ButcherTable::BILLINGTON_3_3_2:
            return "Billington 3-3-2: 2nd order, 3 stages (Implicit)";
        case ButcherTable::TRBDF2_3_3_2:
            return "TR-BDF2 3-3-2: 2nd order, 3 stages (Implicit)";
        case ButcherTable::KVAERNO_4_2_3:
            return "Kvaerno 4-2-3: 3rd order, 4 stages (Implicit)";
        case ButcherTable::ARK324L2SA_DIRK_4_2_3:
            return "ARK3(2)4L[2]SA-DIRK: 3rd order, 4 stages (Implicit)";
        case ButcherTable::ESDIRK436L2SA_6_3_4:
            return "ESDIRK4(3)6L[2]SA-6-3-4: 3rd order, 6 stages (Implicit)";
        case ButcherTable::ESDIRK43I6L2SA_6_3_4:
            return "ESDIRK4(3)I6L[2]SA-6-3-4: 3rd order, 6 stages (Implicit)";
        case ButcherTable::QESDIRK436L2SA_6_3_4:
            return "QESDIRK4(3)6L[2]SA-6-3-4: 3rd order, 6 stages (Implicit)";
        case ButcherTable::CASH_5_2_4:
            return "Cash 5-2-4: 4th order, 5 stages (Implicit)";
        case ButcherTable::CASH_5_3_4:
            return "Cash 5-3-4: 4th order, 5 stages (Implicit)";
        case ButcherTable::SDIRK_5_3_4:
            return "SDIRK 5-3-4: 4th order, 5 stages (Implicit)";
        case ButcherTable::KVAERNO_5_3_4:
            return "Kvaerno 5-3-4: 4th order, 5 stages (Implicit)";
        case ButcherTable::ARK436L2SA_DIRK_6_3_4:
            return "ARK4(3)6L[2]SA-DIRK: 4th order, 6 stages (Implicit)";
        case ButcherTable::ESDIRK437L2SA_7_3_4:
            return "ESDIRK4(3)7L[2]SA-7-3-4: 3rd order, 7 stages (Implicit)";
        case ButcherTable::ARK437L2SA_DIRK_7_3_4:
            return "ARK4(3)7L[2]SA-DIRK: 4th order, 7 stages (Implicit)";
        case ButcherTable::ESDIRK547L2SA2_7_4_5:
            return "ESDIRK5(4)7L[2]SA2-7-4-5: 5th order, 7 stages (Implicit)";
        case ButcherTable::KVAERNO_7_4_5:
            return "Kvaerno 7-4-5: 5th order, 7 stages (Implicit)";
        case ButcherTable::ARK548L2SA_DIRK_8_4_5:
            return "ARK5(4)8L[2]SA-DIRK: 5th order, 8 stages (Implicit)";
        
        case ButcherTable::ARK324L2SA_ERK_4_2_3_DIRK_4_2_3:
            return "ARK3(2)4L[2]SA-ERK-DIRK: 3rd order, 4 stages (ImEx Pair)";
        case ButcherTable::ARK436L2SA_ERK_6_3_4_DIRK_6_3_4:
            return "ARK4(3)6L[2]SA-ERK-DIRK: 4th order, 6 stages (ImEx Pair)";
        case ButcherTable::ARK437L2SA_ERK_7_3_4_DIRK_7_3_4:
            return "ARK4(3)7L[2]SA-ERK-DIRK: 4th order, 7 stages (ImEx Pair)";
        case ButcherTable::ARK548L2SA_ERK_8_4_5_DIRK_8_4_5:
            return "ARK5(4)8L[2]SA-ERK-DIRK: 5th order, 8 stages (ImEx Pair)";
            
        default:
            return "Unknown Butcher table";
    }
}

// Function to check if a Butcher table is explicit
bool is_explicit_method(ButcherTable table) {
    switch (table) {
        case ButcherTable::HEUN_EULER_2_1_2:
        case ButcherTable::BOGACKI_SHAMPINE_4_2_3:
        case ButcherTable::ARK324L2SA_ERK_4_2_3:
        case ButcherTable::ZONNEVELD_5_3_4:
        case ButcherTable::ARK436L2SA_ERK_6_3_4:
        case ButcherTable::ARK437L2SA_ERK_7_3_4:
        case ButcherTable::ARK548L2SA_ERK_8_4_5:
        case ButcherTable::VERNER_8_5_6:
        case ButcherTable::FEHLBERG_13_7_8:
            return true;
        default:
            return false;
    }
}

// Function to check if a Butcher table is implicit
bool is_implicit_method(ButcherTable table) {
    switch (table) {
        case ButcherTable::BACKWARD_EULER_1_1:
        case ButcherTable::ARK2_DIRK_3_1_2:
        case ButcherTable::SDIRK_2_1_2:
        case ButcherTable::IMPLICIT_MIDPOINT_1_2:
        case ButcherTable::IMPLICIT_TRAPEZOIDAL_2_2:
        case ButcherTable::ESDIRK325L2SA_5_2_3:
        case ButcherTable::ESDIRK324L2SA_4_2_3:
        case ButcherTable::ESDIRK32I5L2SA_5_2_3:
        case ButcherTable::BILLINGTON_3_3_2:
        case ButcherTable::TRBDF2_3_3_2:
        case ButcherTable::KVAERNO_4_2_3:
        case ButcherTable::ARK324L2SA_DIRK_4_2_3:
        case ButcherTable::ESDIRK436L2SA_6_3_4:
        case ButcherTable::ESDIRK43I6L2SA_6_3_4:
        case ButcherTable::QESDIRK436L2SA_6_3_4:
        case ButcherTable::CASH_5_2_4:
        case ButcherTable::CASH_5_3_4:
        case ButcherTable::SDIRK_5_3_4:
        case ButcherTable::KVAERNO_5_3_4:
        case ButcherTable::ARK436L2SA_DIRK_6_3_4:
        case ButcherTable::ESDIRK437L2SA_7_3_4:
        case ButcherTable::ARK437L2SA_DIRK_7_3_4:
        case ButcherTable::ESDIRK547L2SA2_7_4_5:
        case ButcherTable::KVAERNO_7_4_5:
        case ButcherTable::ARK548L2SA_DIRK_8_4_5:
            return true;
        default:
            return false;
    }
}

} // namespace sundials_py