cmake_minimum_required(VERSION 3.10)
project(sundials_py)

# Set C++17 as required standard (upgrade from C++11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Function to provide helpful error messages
function(provide_installation_help)
    message(FATAL_ERROR "
========================================================================
SUNDIALS NOT FOUND!

To install SUNDIALS, please use one of the following methods:

1. Using conda (recommended):
   conda install -c conda-forge sundials

2. Using apt (Ubuntu/Debian):
   sudo apt-get install libsundials-dev

3. Using brew (macOS):
   brew install sundials

4. Build from source:
   See https://sundials.readthedocs.io/en/latest/Install_link.html

After installation, please retry: pip install -e .
========================================================================
")
endfunction()

# Add multiple potential paths for finding packages
list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/Library")
list(APPEND CMAKE_PREFIX_PATH "/usr/local")
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")

# Try to detect SUNDIALS installation
set(SUNDIALS_ROOT "")
if(DEFINED ENV{CONDA_PREFIX})
    set(SUNDIALS_ROOT "$ENV{CONDA_PREFIX}")
    set(SUNDIALS_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")
    set(SUNDIALS_LIBRARY_DIR "$ENV{CONDA_PREFIX}/lib")
elseif(EXISTS "/usr/local/include/sundials")
    set(SUNDIALS_ROOT "/usr/local")
    set(SUNDIALS_INCLUDE_DIR "/usr/local/include")
    set(SUNDIALS_LIBRARY_DIR "/usr/local/lib")
elseif(EXISTS "/opt/homebrew/include/sundials")
    set(SUNDIALS_ROOT "/opt/homebrew")
    set(SUNDIALS_INCLUDE_DIR "/opt/homebrew/include")
    set(SUNDIALS_LIBRARY_DIR "/opt/homebrew/lib")
elseif(EXISTS "/usr/include/sundials")
    set(SUNDIALS_ROOT "/usr")
    set(SUNDIALS_INCLUDE_DIR "/usr/include")
    set(SUNDIALS_LIBRARY_DIR "/usr/lib")
endif()

# Print debugging information
message(STATUS "SUNDIALS ROOT: ${SUNDIALS_ROOT}")
message(STATUS "SUNDIALS INCLUDE DIR: ${SUNDIALS_INCLUDE_DIR}")
message(STATUS "SUNDIALS LIBRARY DIR: ${SUNDIALS_LIBRARY_DIR}")

# Check if SUNDIALS headers exist
if(NOT EXISTS "${SUNDIALS_INCLUDE_DIR}/sundials/sundials_config.h")
    provide_installation_help()
endif()

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find pybind11 - try multiple methods
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    # Try to find pybind11 via Python
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    
    if(PYBIND11_CMAKE_DIR)
        list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
        find_package(pybind11 REQUIRED)
    else()
        message(FATAL_ERROR "
========================================================================
PYBIND11 NOT FOUND!

Please install pybind11 with:
    pip install pybind11[global]

After installation, please retry: pip install -e .
========================================================================
")
    endif()
endif()

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

# Load all SUNDIALS libraries for Linux (.so) or Windows (.lib/.dll)
file(GLOB SUNDIALS_LIBS 
    "${SUNDIALS_LIBRARY_DIR}/libsundials_*.so*"
    "${SUNDIALS_LIBRARY_DIR}/sundials_*.lib"
    "${SUNDIALS_LIBRARY_DIR}/libsundials_*.dylib*"
)

# Display found libraries for debugging
message(STATUS "SUNDIALS CVODE library: ${SUNDIALS_CVODE_LIB}")
message(STATUS "SUNDIALS NVEC library: ${SUNDIALS_NVEC_LIB}")
message(STATUS "SUNDIALS ARKODE library: ${SUNDIALS_ARKODE_LIB}")
message(STATUS "SUNDIALS SUNMATRIXDENSE library: ${SUNDIALS_SUNMATRIXDENSE_LIB}")
message(STATUS "SUNDIALS SUNLINSOLDENSE library: ${SUNDIALS_SUNLINSOLDENSE_LIB}")

# Set up SUNDIALS libraries - use the glob result if specific libraries not found
if(SUNDIALS_CVODE_LIB AND SUNDIALS_NVEC_LIB)
    set(SUNDIALS_LIBRARIES 
        ${SUNDIALS_CVODE_LIB}
        ${SUNDIALS_ARKODE_LIB}
        ${SUNDIALS_NVEC_LIB}
        ${SUNDIALS_SUNMATRIXDENSE_LIB}
        ${SUNDIALS_SUNLINSOLDENSE_LIB}
    )
else()
    message(STATUS "Using all found SUNDIALS libraries")
    set(SUNDIALS_LIBRARIES ${SUNDIALS_LIBS})
endif()

# Check if we found any SUNDIALS libraries
if(NOT SUNDIALS_LIBRARIES)
    provide_installation_help()
endif()

message(STATUS "Final SUNDIALS libraries: ${SUNDIALS_LIBRARIES}")

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

# Ensure proper RPATH for finding shared libraries
set_target_properties(_sundials_py PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Install the module
install(TARGETS _sundials_py DESTINATION sundials_py)
