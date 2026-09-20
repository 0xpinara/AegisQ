#include "aegisq/communication_profiler.hpp"

namespace aegisq {

void CommunicationProfiler::record_exchange(OpCode opcode, std::size_t bytes_sent,
                                            std::size_t bytes_received, double seconds) {
    ++metrics_.send_calls;
    ++metrics_.receive_calls;
    ++metrics_.pairwise_exchanges;
    metrics_.bytes_sent += bytes_sent;
    metrics_.bytes_received += bytes_received;
    metrics_.communication_seconds += seconds;

    GateCommunication& entry = metrics_.per_opcode[std::string(opcode_name(opcode))];
    ++entry.exchanges;
    entry.bytes_sent += bytes_sent;
    entry.bytes_received += bytes_received;
    entry.communication_seconds += seconds;
}

void CommunicationProfiler::record_collective(const char* kind, double seconds) {
    const std::string name(kind);
    if (name == "allreduce") {
        ++metrics_.allreduce_calls;
    } else if (name == "allgather") {
        ++metrics_.allgather_calls;
    } else if (name == "barrier") {
        ++metrics_.barrier_calls;
    }
    metrics_.communication_seconds += seconds;
}

void CommunicationProfiler::record_gate(OpCode opcode, double seconds, bool communicated) {
    ++metrics_.gates_applied;
    metrics_.total_seconds += seconds;
    if (communicated) {
        ++metrics_.communicating_gates;
    }
    ++metrics_.per_opcode[std::string(opcode_name(opcode))].gates;

    // Compute time is whatever was not spent communicating. Deriving it this
    // way keeps the two components additive by construction.
    metrics_.compute_seconds = metrics_.total_seconds - metrics_.communication_seconds;
}

void CommunicationProfiler::reset() {
    metrics_ = CommunicationMetrics{};
}

}  // namespace aegisq
