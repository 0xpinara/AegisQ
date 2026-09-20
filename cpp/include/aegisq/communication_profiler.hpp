#pragma once

/// Instrumentation for every byte the runtime moves.
///
/// The project's central claim is about communication volume, so that volume
/// must be *measured*, not modelled. All project-controlled MPI traffic goes
/// through `CommunicationProfiler::record_*`, and the numbers reported by the
/// benchmark suite come from these counters.
///
/// What is counted: payload bytes handed to MPI by AegisQ. What is not
/// counted: MPI's own protocol overhead, and any traffic MPI generates
/// internally for collectives. The distinction is documented rather than
/// papered over.

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>

#include "aegisq/gate.hpp"

namespace aegisq {

/// Traffic attributed to one gate kind.
struct GateCommunication {
    std::uint64_t gates{0};
    std::uint64_t exchanges{0};
    std::uint64_t bytes_sent{0};
    std::uint64_t bytes_received{0};
    double communication_seconds{0.0};
};

struct CommunicationMetrics {
    std::uint64_t send_calls{0};
    std::uint64_t receive_calls{0};
    std::uint64_t pairwise_exchanges{0};

    std::uint64_t bytes_sent{0};
    std::uint64_t bytes_received{0};

    std::uint64_t allreduce_calls{0};
    std::uint64_t allgather_calls{0};
    std::uint64_t barrier_calls{0};

    double communication_seconds{0.0};
    double compute_seconds{0.0};
    double total_seconds{0.0};

    std::uint64_t gates_applied{0};
    std::uint64_t communicating_gates{0};

    /// Breakdown keyed by opcode name.
    std::map<std::string, GateCommunication> per_opcode;
};

class CommunicationProfiler {
  public:
    /// Record one symmetric pairwise exchange (a send and a receive).
    void record_exchange(OpCode opcode, std::size_t bytes_sent, std::size_t bytes_received,
                         double seconds);

    /// Record a collective. `kind` is one of "allreduce", "allgather", "barrier".
    void record_collective(const char* kind, double seconds);

    /// Record that a gate ran, and how long it took in total.
    void record_gate(OpCode opcode, double seconds, bool communicated);

    void reset();

    const CommunicationMetrics& metrics() const { return metrics_; }

  private:
    CommunicationMetrics metrics_;
};

}  // namespace aegisq
