cmake_minimum_required(VERSION 3.10)
project(sundials_py)

# Set C++17 as required standard (upgrade from C++11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add Conda paths for finding packages
list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/Library")

# Provide explicit hints for SUNDIALS location
set(SUNDIALS_ROOT "$ENV{CONDA_PREFIX}")
set(SUNDIALS_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")
set(SUNDIALS_LIBRARY_DIR "$ENV{CONDA_PREFIX}/lib")

# Print debugging information
message(STATUS "SUNDIALS ROOT: ${SUNDIALS_ROOT}")
message(STATUS "SUNDIALS INCLUDE DIR: ${SUNDIALS_INCLUDE_DIR}")
message(STATUS "SUNDIALS LIBRARY DIR: ${SUNDIALS_LIBRARY_DIR}")

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find pybind11
find_package(pybind11 REQUIRED)

# Try to find SUNDIALS libraries with common naming patterns
find_library(SUNDIALS_CVODE_LIB 
    NAMES sundials_cvode libsundials_cvode
    PATHS "${SUNDIALS_LIBRARY_DIR}"
    NO_DEFAULT_PATH)
find_library(SUNDIALS_NVEC_LIB 
    NAMES sundials_nvecserial libsundials_nvecserial
    PATHS "${SUNDIALS_LIBRARY_DIR}"
    NO_DEFAULT_PATH)
find_library(SUNDIALS_ARKODE_LIB 
    NAMES sundials_arkode libsundials_arkode
    PATHS "${SUNDIALS_LIBRARY_DIR}"
    NO_DEFAULT_PATH)
find_library(SUNDIALS_SUNMATRIXDENSE_LIB 
    NAMES sundials_sunmatrixdense libsundials_sunmatrixdense
    PATHS "${SUNDIALS_LIBRARY_DIR}"
    NO_DEFAULT_PATH)
find_library(SUNDIALS_SUNLINSOLDENSE_LIB 
    NAMES sundials_sunlinsoldense libsundials_sunlinsoldense
    PATHS "${SUNDIALS_LIBRARY_DIR}"
    NO_DEFAULT_PATH)
find_library(SUNDIALS_SUNNONLINSOL_LIB
    NAMES sundials_sunnonlinsol libsundials_sunnonlinsol
    PATHS "${SUNDIALS_LIBRARY_DIR}"
    NO_DEFAULT_PATH)

# Load all SUNDIALS libraries for Linux (.so) or Windows (.lib/.dll)
file(GLOB SUNDIALS_LIBS 
    "${SUNDIALS_LIBRARY_DIR}/libsundials_*.so"
    "${SUNDIALS_LIBRARY_DIR}/sundials_*.lib"
)

# Display found libraries for debugging
message(STATUS "SUNDIALS CVODE library: ${SUNDIALS_CVODE_LIB}")
message(STATUS "SUNDIALS NVEC library: ${SUNDIALS_NVEC_LIB}")
message(STATUS "SUNDIALS ARKODE library: ${SUNDIALS_ARKODE_LIB}")
message(STATUS "SUNDIALS SUNMATRIXDENSE library: ${SUNDIALS_SUNMATRIXDENSE_LIB}")
message(STATUS "SUNDIALS SUNLINSOLDENSE library: ${SUNDIALS_SUNLINSOLDENSE_LIB}")
message(STATUS "All SUNDIALS libraries found: ${SUNDIALS_LIBS}")

# Set up SUNDIALS libraries - use the glob result if specific libraries not found
if(SUNDIALS_CVODE_LIB AND SUNDIALS_NVEC_LIB)
    set(SUNDIALS_LIBRARIES 
    ${SUNDIALS_CVODE_LIB}
    ${SUNDIALS_ARKODE_LIB}
    ${SUNDIALS_NVEC_LIB}
    ${SUNDIALS_SUNMATRIXDENSE_LIB}
    ${SUNDIALS_SUNLINSOLDENSE_LIB}
)

# Add SUNNONLINSOL library only if found
if(SUNDIALS_SUNNONLINSOL_LIB)
    list(APPEND SUNDIALS_LIBRARIES ${SUNDIALS_SUNNONLINSOL_LIB})
endif()
else()
    message(STATUS "Using all found SUNDIALS libraries")
    set(SUNDIALS_LIBRARIES ${SUNDIALS_LIBS})
endif()

# Add include directories - both regular include and Library/include
include_directories(
    ${SUNDIALS_INCLUDE_DIR}
    "${SUNDIALS_ROOT}/include"
    "${SUNDIALS_ROOT}/Library/include"
    "${CMAKE_SOURCE_DIR}/src"
    ${Python_INCLUDE_DIRS}
)

# Print Python information for debugging
message(STATUS "Python: ${Python_EXECUTABLE} ${Python_VERSION}")

# Find all source files in the source directory
file(GLOB_RECURSE SOURCES 
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

# Display source files for debugging
message(STATUS "Final sources for Python module: ${SOURCES}")

# Create the Python extension module
pybind11_add_module(_sundials_py ${SOURCES})

# Link against SUNDIALS libraries
target_link_libraries(_sundials_py PRIVATE ${SUNDIALS_LIBRARIES})

# Install the module
install(TARGETS _sundials_py DESTINATION sundials_py)