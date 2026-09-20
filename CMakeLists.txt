cmake_minimum_required(VERSION 3.24)

project(AegisQ
    VERSION 0.1.0
    DESCRIPTION "Communication-aware distributed quantum state-vector simulation"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(AEGISQ_ENABLE_MPI    "Build the distributed MPI runtime"        ON)
option(AEGISQ_ENABLE_OPENMP "Use OpenMP for local state-vector kernels" ON)
option(AEGISQ_BUILD_PYTHON  "Build the pybind11 extension module"      ON)
option(AEGISQ_BUILD_TESTS   "Build the C++ unit tests"                 ON)
option(AEGISQ_NATIVE_ARCH   "Compile with -march=native"               OFF)

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
set(AEGISQ_HAVE_MPI 0)
if(AEGISQ_ENABLE_MPI)
    find_package(MPI COMPONENTS CXX)
    if(MPI_CXX_FOUND)
        set(AEGISQ_HAVE_MPI 1)
        message(STATUS "AegisQ: MPI enabled (${MPI_CXX_COMPILER})")
    else()
        message(WARNING "AegisQ: MPI requested but not found; building single-process only")
    endif()
endif()

# OpenMP: Apple's clang needs libomp wired up by hand, so fall back to a manual
# interface target when FindOpenMP cannot produce one.
add_library(aegisq_openmp INTERFACE)
set(AEGISQ_HAVE_OPENMP 0)
if(AEGISQ_ENABLE_OPENMP)
    find_package(OpenMP COMPONENTS CXX QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(aegisq_openmp INTERFACE OpenMP::OpenMP_CXX)
        set(AEGISQ_HAVE_OPENMP 1)
    else()
        find_path(AEGISQ_LIBOMP_INCLUDE omp.h
            HINTS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
        find_library(AEGISQ_LIBOMP_LIBRARY omp
            HINTS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)
        if(AEGISQ_LIBOMP_INCLUDE AND AEGISQ_LIBOMP_LIBRARY)
            target_compile_options(aegisq_openmp INTERFACE -Xpreprocessor -fopenmp)
            target_include_directories(aegisq_openmp INTERFACE ${AEGISQ_LIBOMP_INCLUDE})
            target_link_libraries(aegisq_openmp INTERFACE ${AEGISQ_LIBOMP_LIBRARY})
            set(AEGISQ_HAVE_OPENMP 1)
        endif()
    endif()
    if(AEGISQ_HAVE_OPENMP)
        message(STATUS "AegisQ: OpenMP enabled")
    else()
        message(STATUS "AegisQ: OpenMP not available; local kernels stay single-threaded")
    endif()
endif()

# ---------------------------------------------------------------------------
# Core library
# ---------------------------------------------------------------------------
set(AEGISQ_SOURCES
    cpp/src/version.cpp
    cpp/src/gates.cpp
    cpp/src/circuit.cpp
    cpp/src/statevector.cpp
    cpp/src/measurement.cpp
    cpp/src/mpi_context.cpp
    cpp/src/distributed_layout.cpp
    cpp/src/communication_profiler.cpp
    cpp/src/bandwidth.cpp
    cpp/src/distributed_gates.cpp
)

add_library(aegisq_core STATIC ${AEGISQ_SOURCES})
add_library(AegisQ::core ALIAS aegisq_core)

target_include_directories(aegisq_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>)

target_compile_definitions(aegisq_core PUBLIC
    AEGISQ_HAVE_MPI=${AEGISQ_HAVE_MPI}
    AEGISQ_HAVE_OPENMP=${AEGISQ_HAVE_OPENMP}
    AEGISQ_VERSION_STRING="${PROJECT_VERSION}"
    AEGISQ_COMPILER_STRING="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

target_link_libraries(aegisq_core PUBLIC aegisq_openmp)
if(AEGISQ_HAVE_MPI)
    target_link_libraries(aegisq_core PUBLIC MPI::MPI_CXX)
endif()

target_compile_options(aegisq_core PRIVATE -Wall -Wextra -Wpedantic)
if(AEGISQ_NATIVE_ARCH)
    target_compile_options(aegisq_core PUBLIC -march=native)
endif()

# ---------------------------------------------------------------------------
# Python extension
# ---------------------------------------------------------------------------
if(AEGISQ_BUILD_PYTHON)
    find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
    if(NOT pybind11_DIR)
        execute_process(
            COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE AEGISQ_PYBIND11_PROBE)
        if(NOT AEGISQ_PYBIND11_PROBE EQUAL 0)
            message(FATAL_ERROR "pybind11 not importable by ${Python_EXECUTABLE}")
        endif()
    endif()
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_aegisq_core cpp/src/bindings.cpp)
    target_link_libraries(_aegisq_core PRIVATE aegisq_core)

    # Drop the extension straight into the source package so that both an
    # editable install and an in-tree `pytest` run pick it up.
    set_target_properties(_aegisq_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/aegisq"
        LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_SOURCE_DIR}/aegisq"
        LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_SOURCE_DIR}/aegisq")
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(AEGISQ_BUILD_TESTS)
    enable_testing()
    add_subdirectory(cpp/tests)
endif()

message(STATUS "AegisQ ${PROJECT_VERSION}: MPI=${AEGISQ_HAVE_MPI} OpenMP=${AEGISQ_HAVE_OPENMP} "
               "python=${AEGISQ_BUILD_PYTHON} tests=${AEGISQ_BUILD_TESTS}")
