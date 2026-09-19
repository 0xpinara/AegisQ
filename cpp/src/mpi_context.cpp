#include "aegisq/mpi_context.hpp"

#include <cstdlib>
#include <stdexcept>

namespace aegisq {
namespace {

bool g_finalized = false;

}  // namespace

MpiContext::MpiContext() {
#if AEGISQ_HAVE_MPI
    int already_initialized = 0;
    MPI_Initialized(&already_initialized);
    if (!already_initialized) {
        // MPI_THREAD_FUNNELED: only the thread that called MPI_Init issues MPI
        // calls, which is exactly how the OpenMP kernels are structured (all
        // communication happens outside parallel regions).
        int provided = 0;
        const int status = MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided);
        if (status != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Init_thread failed");
        }
        std::atexit(&MpiContext::finalize);
    }
    MPI_Comm_rank(comm_, &rank_);
    MPI_Comm_size(comm_, &world_size_);
    initialized_ = true;
#else
    rank_ = 0;
    world_size_ = 1;
    initialized_ = false;
#endif
}

MpiContext& MpiContext::instance() {
    static MpiContext context;
    return context;
}

bool MpiContext::compiled_with_mpi() {
#if AEGISQ_HAVE_MPI
    return true;
#else
    return false;
#endif
}

std::string MpiContext::library_version() {
#if AEGISQ_HAVE_MPI
    char buffer[MPI_MAX_LIBRARY_VERSION_STRING];
    int length = 0;
    if (MPI_Get_library_version(buffer, &length) == MPI_SUCCESS && length > 0) {
        std::string version(buffer, static_cast<std::size_t>(length));
        const std::size_t newline = version.find('\n');
        return newline == std::string::npos ? version : version.substr(0, newline);
    }
    return "unknown MPI";
#else
    return "none";
#endif
}

void MpiContext::barrier() const {
#if AEGISQ_HAVE_MPI
    if (initialized_ && !g_finalized) {
        MPI_Barrier(comm_);
    }
#endif
}

void MpiContext::finalize() {
#if AEGISQ_HAVE_MPI
    if (g_finalized) {
        return;
    }
    int initialized = 0;
    int finalized = 0;
    MPI_Initialized(&initialized);
    MPI_Finalized(&finalized);
    if (initialized && !finalized) {
        MPI_Finalize();
    }
    g_finalized = true;
#endif
}

}  // namespace aegisq
