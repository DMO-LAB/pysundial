cmake_minimum_required(VERSION 3.10)
project(SundialsPy)

# Set C++17 as required standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Platform-specific paths
if(DEFINED ENV{CONDA_PREFIX})
    # Conda environment (your current setup)
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}/Library")
    set(SUNDIALS_ROOT "$ENV{CONDA_PREFIX}")
    set(SUNDIALS_INCLUDE_DIR "$ENV{CONDA_PREFIX}/include")
    set(SUNDIALS_LIBRARY_DIR "$ENV{CONDA_PREFIX}/lib")
    message(STATUS "Using Conda environment")
elseif(APPLE)
    # macOS with Homebrew
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew" "/usr/local")
    set(SUNDIALS_ROOT "/opt/homebrew")
    set(SUNDIALS_INCLUDE_DIR "/opt/homebrew/include")
    set(SUNDIALS_LIBRARY_DIR "/opt/homebrew/lib")
    message(STATUS "Using macOS Homebrew paths")
elseif(WIN32)
    # Windows
    set(SUNDIALS_ROOT "C:/sundials")
    set(SUNDIALS_INCLUDE_DIR "C:/sundials/include")
    set(SUNDIALS_LIBRARY_DIR "C:/sundials/lib")
    message(STATUS "Using Windows paths")
else()
    # Linux
    set(SUNDIALS_ROOT "/usr/local")
    set(SUNDIALS_INCLUDE_DIR "/usr/local/include")
    set(SUNDIALS_LIBRARY_DIR "/usr/local/lib")
    message(STATUS "Using Linux paths")
endif()

# Print debugging information
message(STATUS "SUNDIALS ROOT: ${SUNDIALS_ROOT}")
message(STATUS "SUNDIALS INCLUDE DIR: ${SUNDIALS_INCLUDE_DIR}")
message(STATUS "SUNDIALS LIBRARY DIR: ${SUNDIALS_LIBRARY_DIR}")

# Check if SUNDIALS headers exist
if(NOT EXISTS "${SUNDIALS_INCLUDE_DIR}/sundials/sundials_config.h")
    message(FATAL_ERROR
        "SUNDIALS headers not found at ${SUNDIALS_INCLUDE_DIR}/sundials/sundials_config.h\n"
        "Please install SUNDIALS:\n"
        "  Conda:  conda install -c conda-forge sundials\n"
        "  macOS:  brew install sundials\n"
        "  Ubuntu: sudo apt-get install libsundials-dev\n"
        "Or set SUNDIALS_ROOT, SUNDIALS_INCLUDE_DIR, SUNDIALS_LIBRARY_DIR manually."
    )
endif()

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Find pybind11 - try Python method first (more reliable)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_RESULT
    ERROR_QUIET
)

if(PYBIND11_RESULT EQUAL 0 AND PYBIND11_CMAKE_DIR)
    message(STATUS "Found pybind11 via Python: ${PYBIND11_CMAKE_DIR}")
    list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_DIR})
    find_package(pybind11 REQUIRED)
else()
    # Fallback to find_package
    find_package(pybind11 REQUIRED)
endif()

# Try to find SUNDIALS libraries with platform-specific naming
if(WIN32)
    # Windows library names
    find_library(SUNDIALS_CVODE_LIB NAMES sundials_cvode libsundials_cvode PATHS "${SUNDIALS_LIBRARY_DIR}")
    find_library(SUNDIALS_NVEC_LIB NAMES sundials_nvecserial libsundials_nvecserial PATHS "${SUNDIALS_LIBRARY_DIR}")
    find_library(SUNDIALS_ARKODE_LIB NAMES sundials_arkode libsundials_arkode PATHS "${SUNDIALS_LIBRARY_DIR}")
    find_library(SUNDIALS_SUNMATRIXDENSE_LIB NAMES sundials_sunmatrixdense libsundials_sunmatrixdense PATHS "${SUNDIALS_LIBRARY_DIR}")
    find_library(SUNDIALS_SUNLINSOLDENSE_LIB NAMES sundials_sunlinsoldense libsundials_sunlinsoldense PATHS "${SUNDIALS_LIBRARY_DIR}")
    find_library(SUNDIALS_SUNNONLINSOL_LIB NAMES sundials_sunnonlinsol libsundials_sunnonlinsol PATHS "${SUNDIALS_LIBRARY_DIR}")
else()
    # Unix library names (your current working setup)
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
endif()

# Display found libraries for debugging
message(STATUS "SUNDIALS CVODE library: ${SUNDIALS_CVODE_LIB}")
message(STATUS "SUNDIALS NVEC library: ${SUNDIALS_NVEC_LIB}")
message(STATUS "SUNDIALS ARKODE library: ${SUNDIALS_ARKODE_LIB}")
message(STATUS "SUNDIALS SUNMATRIXDENSE library: ${SUNDIALS_SUNMATRIXDENSE_LIB}")
message(STATUS "SUNDIALS SUNLINSOLDENSE library: ${SUNDIALS_SUNLINSOLDENSE_LIB}")

# Set up SUNDIALS libraries
if(SUNDIALS_CVODE_LIB AND SUNDIALS_NVEC_LIB)
    set(SUNDIALS_LIBRARIES 
        ${SUNDIALS_CVODE_LIB}
        ${SUNDIALS_ARKODE_LIB}
        ${SUNDIALS_NVEC_LIB}
        ${SUNDIALS_SUNMATRIXDENSE_LIB}
        ${SUNDIALS_SUNLINSOLDENSE_LIB}
    )
    
    if(SUNDIALS_SUNNONLINSOL_LIB)
        list(APPEND SUNDIALS_LIBRARIES ${SUNDIALS_SUNNONLINSOL_LIB})
    endif()
else()
    message(FATAL_ERROR "Required SUNDIALS libraries not found")
endif()

# Add include directories
include_directories(
    ${SUNDIALS_INCLUDE_DIR}
    "${SUNDIALS_ROOT}/include"
    "${SUNDIALS_ROOT}/Library/include"
    "${CMAKE_SOURCE_DIR}/src"
    ${Python_INCLUDE_DIRS}
)

# Find all source files
file(GLOB_RECURSE SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

# Create the Python extension module (with underscore for correct naming)
pybind11_add_module(_SundialsPy ${SOURCES})

# Link against SUNDIALS libraries
target_link_libraries(_SundialsPy PRIVATE ${SUNDIALS_LIBRARIES})

# Ensure proper RPATH for finding shared libraries
set_target_properties(_SundialsPy PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Install the module
install(TARGETS _SundialsPy DESTINATION SundialsPy)
