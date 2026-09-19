#pragma once

/// Thin RAII-free wrapper around the MPI world communicator.
///
/// MPI is initialised lazily on first use rather than at module import, so
/// `import aegisq` in an ordinary Python process stays cheap and side-effect
/// free. When the core is compiled without MPI the context degrades to a
/// single-rank world, which lets the distributed code paths be exercised
/// (and unit tested) in a serial build.

#include <cstdint>
#include <string>

#if AEGISQ_HAVE_MPI
#include <mpi.h>
#endif

namespace aegisq {

class MpiContext {
  public:
    /// Lazily initialise MPI (if compiled in) and return the process-wide context.
    static MpiContext& instance();

    /// True when the core was compiled against an MPI implementation.
    static bool compiled_with_mpi();

    /// Implementation and version string, or "none" for a serial build.
    static std::string library_version();

    /// Shut MPI down. Safe to call more than once; also registered with atexit.
    static void finalize();

    bool initialized() const { return initialized_; }
    int rank() const { return rank_; }
    int world_size() const { return world_size_; }

    void barrier() const;

#if AEGISQ_HAVE_MPI
    MPI_Comm comm() const { return comm_; }
#endif

  private:
    MpiContext();

    int rank_{0};
    int world_size_{1};
    bool initialized_{false};
#if AEGISQ_HAVE_MPI
    MPI_Comm comm_{MPI_COMM_WORLD};
#endif
};

}  // namespace aegisq
