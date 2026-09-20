/// Distributed state-vector construction and gate execution.
///
/// Everything here is driven by `DistributedLayout`: whether a gate needs
/// communication at all is a property of where its operands sit, not of the
/// gate alone.

#include <algorithm>
#include <chrono>
#include <stdexcept>

#include "aegisq/kernels.hpp"
#include "aegisq/statevector.hpp"

namespace aegisq {

template <typename Real>
DistributedStateVectorT<Real>::DistributedStateVectorT(int num_qubits)
    : DistributedStateVectorT(num_qubits, std::vector<int>{}) {}

template <typename Real>
DistributedStateVectorT<Real>::DistributedStateVectorT(int num_qubits,
                                                       std::vector<int> logical_to_position)
    : layout_(num_qubits, MpiContext::instance().world_size(), MpiContext::instance().rank(),
              std::move(logical_to_position)),
      local_(layout_.local_state_size(), Amplitude{0, 0}) {
    reset();
}

template <typename Real>
void DistributedStateVectorT<Real>::reset() {
    std::fill(local_.begin(), local_.end(), Amplitude{0, 0});
    // |0...0> is physical index 0 regardless of the qubit mapping, and index 0
    // lives on rank 0 by construction.
    if (layout_.rank() == 0) {
        local_[0] = Amplitude{1, 0};
    }
    profiler_.reset();
}

template <typename Real>
double DistributedStateVectorT<Real>::local_squared_norm() const {
    return kernels::squared_norm(local_.data(), local_.size());
}

template <typename Real>
double DistributedStateVectorT<Real>::norm() const {
    double local_total = local_squared_norm();
#if AEGISQ_HAVE_MPI
    if (layout_.world_size() > 1) {
        const auto started = std::chrono::steady_clock::now();
        double global_total = 0.0;
        MPI_Allreduce(&local_total, &global_total, 1, MPI_DOUBLE, MPI_SUM,
                      MpiContext::instance().comm());
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - started;
        // `norm` is logically const but the profiler is a measurement sink.
        const_cast<CommunicationProfiler&>(profiler_).record_collective("allreduce",
                                                                        elapsed.count());
        return global_total;
    }
#endif
    return local_total;
}

template <typename Real>
std::vector<std::complex<double>> DistributedStateVectorT<Real>::gather() const {
    const std::size_t local_size = local_.size();
    std::vector<std::complex<double>> physical(local_size *
                                               static_cast<std::size_t>(layout_.world_size()));

    std::vector<std::complex<double>> local_as_double(local_size);
    for (std::size_t i = 0; i < local_size; ++i) {
        local_as_double[i] = std::complex<double>(static_cast<double>(local_[i].real()),
                                                  static_cast<double>(local_[i].imag()));
    }

#if AEGISQ_HAVE_MPI
    if (layout_.world_size() > 1) {
        const auto started = std::chrono::steady_clock::now();
        MPI_Allgather(local_as_double.data(), static_cast<int>(local_size), MPI_C_DOUBLE_COMPLEX,
                      physical.data(), static_cast<int>(local_size), MPI_C_DOUBLE_COMPLEX,
                      MpiContext::instance().comm());
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - started;
        const_cast<CommunicationProfiler&>(profiler_).record_collective("allgather",
                                                                        elapsed.count());
    } else {
        physical = local_as_double;
    }
#else
    physical = local_as_double;
#endif

    if (layout_.is_identity_mapping()) {
        return physical;
    }

    // Re-order into logical basis order so callers can compare against the
    // single-process reference without knowing the placement.
    std::vector<std::complex<double>> logical(physical.size());
    for (std::uint64_t index = 0; index < physical.size(); ++index) {
        logical[layout_.to_logical_index(index)] = physical[index];
    }
    return logical;
}

// ---------------------------------------------------------------------------
// Gate execution
// ---------------------------------------------------------------------------

template <typename Real>
void DistributedStateVectorT<Real>::check_operands(const Gate& gate) const {
    const int arity = gate_arity(gate.opcode);
    for (int i = 0; i < arity; ++i) {
        const int q = gate.qubits[static_cast<std::size_t>(i)];
        if (q < 0 || q >= layout_.num_qubits()) {
            throw std::invalid_argument("qubit index out of range in " + to_string(gate));
        }
    }
    if (arity == 2 && gate.qubits[0] == gate.qubits[1]) {
        throw std::invalid_argument("two-qubit gate needs distinct operands: " + to_string(gate));
    }
}

template <typename Real>
void DistributedStateVectorT<Real>::apply_gate(const Gate& gate) {
    check_operands(gate);

    current_opcode_ = gate.opcode;
    current_gate_communicated_ = false;
    const auto gate_started = std::chrono::steady_clock::now();

    Amplitude* psi = local_.data();
    const std::size_t n = local_.size();

    switch (gate.opcode) {
        case OpCode::X:
        case OpCode::Y:
        case OpCode::H:
        case OpCode::RX:
        case OpCode::RY: {
            const int q = gate.qubits[0];
            if (layout_.is_local(q)) {
                kernels::apply_single_qubit(psi, n, layout_.position(q), single_qubit_matrix(gate));
            } else {
                apply_global_single_qubit(gate);
            }
            break;
        }

        case OpCode::Z:
        case OpCode::S:
        case OpCode::T:
        case OpCode::RZ: {
            const int q = gate.qubits[0];
            const auto diagonal = diagonal_entries(gate);
            if (layout_.is_local(q)) {
                kernels::apply_diagonal(psi, n, layout_.position(q), diagonal);
            } else {
                // The rank id fixes this qubit's value for every amplitude on
                // the rank, so the gate degenerates to one scalar multiply.
                kernels::scale_all(psi, n,
                                   diagonal[static_cast<std::size_t>(layout_.global_bit(q))]);
            }
            break;
        }

        case OpCode::CZ: {
            const int a = gate.qubits[0];
            const int b = gate.qubits[1];
            const bool a_local = layout_.is_local(a);
            const bool b_local = layout_.is_local(b);
            if (a_local && b_local) {
                kernels::apply_cz(psi, n, layout_.position(a), layout_.position(b));
            } else if (a_local != b_local) {
                const int global_qubit = a_local ? b : a;
                const int local_qubit = a_local ? a : b;
                if (layout_.global_bit(global_qubit) == 1) {
                    kernels::apply_phase_if_bit_set(psi, n, layout_.position(local_qubit), -1.0);
                }
            } else if (layout_.global_bit(a) == 1 && layout_.global_bit(b) == 1) {
                kernels::scale_all(psi, n, -1.0);
            }
            break;
        }

        case OpCode::CX: {
            const int control = gate.qubits[0];
            const int target = gate.qubits[1];
            if (layout_.is_local(target)) {
                const auto x_matrix = single_qubit_matrix(Gate::one(OpCode::X, 0));
                if (layout_.is_local(control)) {
                    kernels::apply_controlled_single_qubit(psi, n, layout_.position(control),
                                                           layout_.position(target), x_matrix);
                } else if (layout_.global_bit(control) == 1) {
                    // The rank already satisfies the control: flip locally.
                    kernels::apply_single_qubit(psi, n, layout_.position(target), x_matrix);
                }
            } else {
                apply_cnot_global_target(control, target);
            }
            break;
        }

        case OpCode::SWAP: {
            const int a = gate.qubits[0];
            const int b = gate.qubits[1];
            if (layout_.is_local(a) && layout_.is_local(b)) {
                kernels::apply_swap(psi, n, layout_.position(a), layout_.position(b));
            } else {
                apply_swap_with_global(a, b);
            }
            break;
        }
    }

    const std::chrono::duration<double> gate_elapsed =
        std::chrono::steady_clock::now() - gate_started;
    profiler_.record_gate(gate.opcode, gate_elapsed.count(), current_gate_communicated_);
}

template <typename Real>
void DistributedStateVectorT<Real>::apply_circuit(const Circuit& circuit) {
    if (circuit.num_qubits() != layout_.num_qubits()) {
        throw std::invalid_argument("circuit width does not match the distributed state");
    }
    for (const Gate& gate : circuit.gates()) {
        apply_gate(gate);
    }
}

template <typename Real>
void DistributedStateVectorT<Real>::apply_x(int qubit) {
    apply_gate(Gate::one(OpCode::X, qubit));
}
template <typename Real>
void DistributedStateVectorT<Real>::apply_y(int qubit) {
    apply_gate(Gate::one(OpCode::Y, qubit));
}
template <typename Real>
void DistributedStateVectorT<Real>::apply_z(int qubit) {
    apply_gate(Gate::one(OpCode::Z, qubit));
}
template <typename Real>
void DistributedStateVectorT<Real>::apply_h(int qubit) {
    apply_gate(Gate::one(OpCode::H, qubit));
}
template <typename Real>
void DistributedStateVectorT<Real>::apply_s(int qubit) {
    apply_gate(Gate::one(OpCode::S, qubit));
}
template <typename Real>
void DistributedStateVectorT<Real>::apply_t(int qubit) {
    apply_gate(Gate::one(OpCode::T, qubit));
}
template <typename Real>
void DistributedStateVectorT<Real>::apply_rx(int qubit, double theta) {
    apply_gate(Gate::one(OpCode::RX, qubit, theta));
}
template <typename Real>
void DistributedStateVectorT<Real>::apply_ry(int qubit, double theta) {
    apply_gate(Gate::one(OpCode::RY, qubit, theta));
}
template <typename Real>
void DistributedStateVectorT<Real>::apply_rz(int qubit, double theta) {
    apply_gate(Gate::one(OpCode::RZ, qubit, theta));
}
template <typename Real>
void DistributedStateVectorT<Real>::apply_cnot(int control, int target) {
    apply_gate(Gate::two(OpCode::CX, control, target));
}
template <typename Real>
void DistributedStateVectorT<Real>::apply_cz(int a, int b) {
    apply_gate(Gate::two(OpCode::CZ, a, b));
}
template <typename Real>
void DistributedStateVectorT<Real>::apply_swap(int a, int b) {
    apply_gate(Gate::two(OpCode::SWAP, a, b));
}

// ---------------------------------------------------------------------------
// Operations that still need a pairwise exchange (implemented in later phases)
// ---------------------------------------------------------------------------

template <typename Real>
std::vector<std::complex<Real>>& DistributedStateVectorT<Real>::exchange_buffer(std::size_t count) {
    if (exchange_.size() < count) {
        exchange_.resize(count);
    }
    return exchange_;
}

template <typename Real>
void DistributedStateVectorT<Real>::exchange_with_partner(int partner, const Amplitude* send,
                                                          Amplitude* receive, std::size_t count) {
#if AEGISQ_HAVE_MPI
    const auto started = std::chrono::steady_clock::now();
    // MPI element counts are int-typed. Shards larger than that are split into
    // chunks rather than silently overflowing.
    constexpr std::size_t kMaxChunk = 1ULL << 28;
    std::size_t offset = 0;
    while (offset < count) {
        const std::size_t chunk = std::min(kMaxChunk, count - offset);
        const int status = MPI_Sendrecv(
            send + offset, static_cast<int>(chunk), MpiAmplitudeType<Real>::value(), partner, 0,
            receive + offset, static_cast<int>(chunk), MpiAmplitudeType<Real>::value(), partner, 0,
            MpiContext::instance().comm(), MPI_STATUS_IGNORE);
        if (status != MPI_SUCCESS) {
            throw std::runtime_error("MPI_Sendrecv failed during a shard exchange");
        }
        offset += chunk;
    }

    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - started;
    const std::size_t bytes = count * MpiAmplitudeType<Real>::bytes;
    profiler_.record_exchange(current_opcode_, bytes, bytes, elapsed.count());
    current_gate_communicated_ = true;
#else
    (void)partner;
    (void)send;
    (void)receive;
    (void)count;
    throw std::runtime_error("a pairwise exchange was requested in a build without MPI");
#endif
}

template <typename Real>
void DistributedStateVectorT<Real>::apply_global_single_qubit(const Gate& gate) {
    const int qubit = gate.qubits[0];
    const int partner = layout_.partner_rank_for_global_qubit(qubit);
    const int bit = layout_.global_bit(qubit);
    const std::size_t count = local_.size();

    // Every amplitude on this rank pairs with one on the partner rank, so the
    // whole shard crosses the network exactly once.
    std::vector<Amplitude>& incoming = exchange_buffer(count);
    exchange_with_partner(partner, local_.data(), incoming.data(), count);

    kernels::apply_single_qubit_paired(local_.data(), incoming.data(), count,
                                       single_qubit_matrix(gate), bit);
}

template <typename Real>
void DistributedStateVectorT<Real>::exchange_full_shard(int partner) {
    const std::size_t count = local_.size();
    std::vector<Amplitude>& incoming = exchange_buffer(count);
    exchange_with_partner(partner, local_.data(), incoming.data(), count);
    kernels::overwrite_from(local_.data(), incoming.data(), count);
}

template <typename Real>
void DistributedStateVectorT<Real>::exchange_half_shard(int partner, int bit, int value) {
    const std::size_t count = local_.size();
    const std::size_t half = kernels::packed_count(count, bit);

    if (packed_.size() < half) {
        packed_.resize(half);
    }
    std::vector<Amplitude>& incoming = exchange_buffer(half);

    // Only the amplitudes selected by (bit, value) participate, so only those
    // cross the network: half the shard instead of all of it.
    kernels::pack_by_bit(local_.data(), packed_.data(), count, bit, value);
    exchange_with_partner(partner, packed_.data(), incoming.data(), half);
    kernels::unpack_by_bit(local_.data(), incoming.data(), count, bit, value);
}

template <typename Real>
void DistributedStateVectorT<Real>::apply_cnot_global_target(int control, int target) {
    const int partner = layout_.partner_rank_for_global_qubit(target);

    if (layout_.is_local(control)) {
        // Case C: control local, target global.
        //
        // A CX moves the amplitude at index x (control bit set) to
        // x XOR target_bit. The target bit selects the rank, so those
        // amplitudes swap with the *same* local index on the partner rank,
        // while amplitudes with a clear control bit stay put. Both ranks hold
        // control-set amplitudes, so the exchange is symmetric.
        exchange_half_shard(partner, layout_.position(control), 1);
        return;
    }

    // Case D: control global, target global.
    //
    // The control bit is fixed per rank. Ranks that do not satisfy it do
    // nothing at all; the rest swap their entire shard with the partner,
    // which also satisfies the control because the two global positions are
    // different bits of the rank id.
    if (layout_.global_bit(control) == 1) {
        exchange_full_shard(partner);
    }
}

template <typename Real>
void DistributedStateVectorT<Real>::apply_swap_with_global(int a, int b) {
    const bool a_local = layout_.is_local(a);
    const bool b_local = layout_.is_local(b);

    if (!a_local && !b_local) {
        // Both global: the swap permutes rank ids only. Ranks whose two bits
        // agree are fixed points; the others exchange whole shards with the
        // rank that has those two bits transposed.
        const int bit_a = layout_.global_bit(a);
        const int bit_b = layout_.global_bit(b);
        if (bit_a == bit_b) {
            return;
        }
        const int partner =
            layout_.rank() ^ (1 << layout_.global_position(a)) ^ (1 << layout_.global_position(b));
        exchange_full_shard(partner);
        return;
    }

    // One local, one global: amplitudes with (global bit, local bit) = (0, 1)
    // trade places with (1, 0). A rank therefore sends the half of its shard
    // whose local bit is the opposite of its own global bit, and receives the
    // partner's complementary half into those same slots.
    const int global_qubit = a_local ? b : a;
    const int local_qubit = a_local ? a : b;
    const int selected_value = layout_.global_bit(global_qubit) == 0 ? 1 : 0;
    const int partner = layout_.partner_rank_for_global_qubit(global_qubit);
    exchange_half_shard(partner, layout_.position(local_qubit), selected_value);
}

template <typename Real>
CommunicationMetrics DistributedStateVectorT<Real>::reduced_metrics() const {
    CommunicationMetrics reduced = profiler_.metrics();
#if AEGISQ_HAVE_MPI
    if (layout_.world_size() > 1) {
        std::uint64_t counters[9] = {
            reduced.send_calls,      reduced.receive_calls,  reduced.pairwise_exchanges,
            reduced.bytes_sent,      reduced.bytes_received, reduced.allreduce_calls,
            reduced.allgather_calls, reduced.barrier_calls,  reduced.communicating_gates};
        std::uint64_t summed[9] = {0};
        MPI_Allreduce(counters, summed, 9, MPI_UINT64_T, MPI_SUM, MpiContext::instance().comm());

        double times[3] = {reduced.communication_seconds, reduced.compute_seconds,
                           reduced.total_seconds};
        double slowest[3] = {0.0, 0.0, 0.0};
        MPI_Allreduce(times, slowest, 3, MPI_DOUBLE, MPI_MAX, MpiContext::instance().comm());

        reduced.send_calls = summed[0];
        reduced.receive_calls = summed[1];
        reduced.pairwise_exchanges = summed[2];
        reduced.bytes_sent = summed[3];
        reduced.bytes_received = summed[4];
        reduced.allreduce_calls = summed[5];
        reduced.allgather_calls = summed[6];
        reduced.barrier_calls = summed[7];
        reduced.communicating_gates = summed[8];
        reduced.communication_seconds = slowest[0];
        reduced.compute_seconds = slowest[1];
        reduced.total_seconds = slowest[2];
    }
#endif
    return reduced;
}

template class DistributedStateVectorT<double>;
template class DistributedStateVectorT<float>;

}  // namespace aegisq
