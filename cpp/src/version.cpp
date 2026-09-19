#include "aegisq/version.hpp"

#if AEGISQ_HAVE_OPENMP
#include <omp.h>
#endif

namespace aegisq {

std::string version() {
    return AEGISQ_VERSION_STRING;
}

std::string compiler() {
    return AEGISQ_COMPILER_STRING;
}

bool has_mpi() {
#if AEGISQ_HAVE_MPI
    return true;
#else
    return false;
#endif
}

bool has_openmp() {
#if AEGISQ_HAVE_OPENMP
    return true;
#else
    return false;
#endif
}

int max_threads() {
#if AEGISQ_HAVE_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

}  // namespace aegisq
