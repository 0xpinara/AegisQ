#pragma once

#include <string>

namespace aegisq {

/// Semantic version of the compiled core library.
std::string version();

/// Compiler identification string captured at configure time.
std::string compiler();

/// True when the core was compiled against an MPI implementation.
bool has_mpi();

/// True when local kernels were compiled with OpenMP support.
bool has_openmp();

/// Number of OpenMP threads the runtime would use for a large local kernel.
int max_threads();

}  // namespace aegisq
