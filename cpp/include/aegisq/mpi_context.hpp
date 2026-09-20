#pragma once

/// Thin RAII-free wrapper around the MPI world communicator.
///
/// MPI is initialised lazily on first use rather than at module import, so
/// `import aegisq` in an ordinary Python process stays cheap and side-effect
/// free. When the core is compiled without MPI the context degrades to a
/// single-rank world, which lets the distributed code paths be exercised
/// (and unit tested) in a serial build.

#include <cstddef>
#include <cstdint>
#include <string>

#if AEGISQ_HAVE_MPI
#include <mpi.h>
#endif

namespace aegisq {

#if AEGISQ_HAVE_MPI
/// MPI datatype for a complex amplitude of a given real type.
template <typename Real>
struct MpiAmplitudeType;

template <>
struct MpiAmplitudeType<double> {
    static MPI_Datatype value() { return MPI_C_DOUBLE_COMPLEX; }
    static constexpr std::size_t bytes = 16;
};

template <>
struct MpiAmplitudeType<float> {
    static MPI_Datatype value() { return MPI_C_FLOAT_COMPLEX; }
    static constexpr std::size_t bytes = 8;
};
#endif

/// Largest number of amplitudes handed to a single MPI call.
///
/// MPI element counts are `int`-typed, so a shard bigger than that must be
/// split. The limit is adjustable at runtime for one reason: at its natural
/// value the splitting path needs a 4 GiB shard to trigger, so it would never
/// be exercised by a test and would rot silently.
std::size_t max_exchange_elements();
void set_max_exchange_elements(std::size_t count);

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
