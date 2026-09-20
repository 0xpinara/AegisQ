#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "aegisq/circuit.hpp"
#include "aegisq/communication_profiler.hpp"
#include "aegisq/distributed_layout.hpp"
#include "aegisq/gate.hpp"
#include "aegisq/measurement.hpp"
#include "aegisq/mpi_context.hpp"

namespace aegisq {

/// Timing and work counters for one single-process run.
struct LocalMetrics {
    std::uint64_t gates_applied{0};
    double compute_seconds{0.0};
};

/// Single-process state vector.
///
/// Templated on the real type so that fp32 shards (half the memory, half the
/// bytes on the wire) can be benchmarked against fp64 without a second
/// implementation. The distributed runtime reuses the same kernels on each
/// rank's shard.
template <typename Real>
class StateVectorT {
  public:
    using Amplitude = std::complex<Real>;

    explicit StateVectorT(int num_qubits);

    int num_qubits() const { return num_qubits_; }
    std::size_t size() const { return amplitudes_.size(); }

    const Amplitude* data() const { return amplitudes_.data(); }
    Amplitude* data() { return amplitudes_.data(); }
    const std::vector<Amplitude>& amplitudes() const { return amplitudes_; }

    /// Reset to |0...0>.
    void reset();
    void set_amplitudes(const std::vector<Amplitude>& values);

    /// Sum of |amplitude|^2; must stay at 1 under unitary evolution.
    double norm() const;

    void apply_x(int qubit);
    void apply_y(int qubit);
    void apply_z(int qubit);
    void apply_h(int qubit);
    void apply_s(int qubit);
    void apply_t(int qubit);
    void apply_rx(int qubit, double theta);
    void apply_ry(int qubit, double theta);
    void apply_rz(int qubit, double theta);
    void apply_cnot(int control, int target);
    void apply_cz(int a, int b);
    void apply_swap(int a, int b);

    void apply_gate(const Gate& gate);
    void apply_circuit(const Circuit& circuit);

    /// Sample `shots` terminal measurements of all qubits.
    MeasurementResult measure_all(std::uint64_t shots, std::uint64_t seed) const;

    /// Outcome probabilities for every basis state.
    std::vector<double> probabilities() const;

    const LocalMetrics& metrics() const { return metrics_; }
    void reset_metrics() { metrics_ = LocalMetrics{}; }

  private:
    void check_qubit(int qubit) const;

    int num_qubits_;
    std::vector<Amplitude> amplitudes_;
    LocalMetrics metrics_{};
};

using StateVector = StateVectorT<double>;
using StateVectorF32 = StateVectorT<float>;

extern template class StateVectorT<double>;
extern template class StateVectorT<float>;

/// State vector partitioned across MPI ranks.
///
/// Each rank owns `2^L` amplitudes of the global `2^n` state and never
/// materialises the whole vector (except in `gather`, which exists only for
/// tests). Gate execution is dispatched on the layout: a gate whose operands
/// all sit on local positions runs through exactly the same kernels as the
/// single-process engine, while a gate touching a global position may require
/// a pairwise exchange with a partner rank.
template <typename Real>
class DistributedStateVectorT {
  public:
    using Amplitude = std::complex<Real>;

    /// Construct over the MPI world communicator with the identity mapping.
    explicit DistributedStateVectorT(int num_qubits);

    /// Construct with an explicit logical-qubit-to-physical-position mapping.
    DistributedStateVectorT(int num_qubits, std::vector<int> logical_to_position);

    const DistributedLayout& layout() const { return layout_; }
    int num_qubits() const { return layout_.num_qubits(); }
    int rank() const { return layout_.rank(); }
    int world_size() const { return layout_.world_size(); }
    std::size_t local_size() const { return local_.size(); }

    const std::vector<Amplitude>& local_amplitudes() const { return local_; }

    /// Reset to |0...0>.
    void reset();

    void apply_x(int qubit);
    void apply_y(int qubit);
    void apply_z(int qubit);
    void apply_h(int qubit);
    void apply_s(int qubit);
    void apply_t(int qubit);
    void apply_rx(int qubit, double theta);
    void apply_ry(int qubit, double theta);
    void apply_rz(int qubit, double theta);
    void apply_cnot(int control, int target);
    void apply_cz(int a, int b);
    void apply_swap(int a, int b);

    /// Execute one gate, dispatching on where its operands are placed.
    void apply_gate(const Gate& gate);
    void apply_circuit(const Circuit& circuit);

    /// Globally reduced squared norm (an MPI_Allreduce when distributed).
    double norm() const;

    /// Sum of |amplitude|^2 held by this rank alone.
    double local_squared_norm() const;

    /// Collect the full state on every rank, ordered by *logical* basis index.
    ///
    /// Only for tests and small debugging runs: the result is 2^n amplitudes
    /// on every rank, which defeats the point of distributing the state.
    std::vector<std::complex<double>> gather() const;

    /// Sample `shots` terminal measurements of all qubits.
    ///
    /// Counts are keyed by *logical* basis index and are identical on every
    /// rank. The draw sequence depends only on (shots, seed), so a run split
    /// over any number of ranks yields the same counts for a given qubit
    /// placement.
    MeasurementResult measure_all(std::uint64_t shots, std::uint64_t seed) const;

    /// Measured communication and timing counters for this rank.
    const CommunicationMetrics& metrics() const { return profiler_.metrics(); }
    void reset_metrics() { profiler_.reset(); }

    /// Counters summed (bytes, calls) or maximised (times) over all ranks.
    ///
    /// Byte totals are summed because the quantity of interest is how much
    /// traffic the job generated; times are maximised because a distributed
    /// run is only as fast as its slowest rank.
    CommunicationMetrics reduced_metrics() const;

  private:
    void check_operands(const Gate& gate) const;

    /// Non-diagonal single-qubit gate on a global qubit: one pairwise exchange.
    void apply_global_single_qubit(const Gate& gate);

    /// Lazily sized scratch buffer for incoming shards; reused across gates so
    /// a deep circuit does not allocate once per global gate.
    std::vector<Amplitude>& exchange_buffer(std::size_t count);

    /// Symmetric shard exchange with `partner`, chunked to stay inside the
    /// int-typed element counts of the MPI interface.
    void exchange_with_partner(int partner, const Amplitude* send, Amplitude* receive,
                               std::size_t count);

    /// CX whose target sits on a global position.
    void apply_cnot_global_target(int control, int target);

    /// SWAP touching at least one global position.
    void apply_swap_with_global(int a, int b);

    /// Exchange the half of the shard selected by `(bit, value)` with
    /// `partner` and write the incoming half back into the same slots.
    void exchange_half_shard(int partner, int bit, int value);

    /// Exchange the whole shard with `partner` and adopt the incoming data.
    void exchange_full_shard(int partner);

    DistributedLayout layout_;
    std::vector<Amplitude> local_;
    std::vector<Amplitude> exchange_;
    std::vector<Amplitude> packed_;
    CommunicationProfiler profiler_;

    /// Opcode currently executing, so an exchange can be attributed to it.
    OpCode current_opcode_{OpCode::X};
    bool current_gate_communicated_{false};
};

using DistributedStateVector = DistributedStateVectorT<double>;

extern template class DistributedStateVectorT<double>;
extern template class DistributedStateVectorT<float>;

}  // namespace aegisq
