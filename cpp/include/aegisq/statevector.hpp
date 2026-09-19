#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "aegisq/circuit.hpp"
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

    /// Globally reduced squared norm (an MPI_Allreduce when distributed).
    double norm() const;

    /// Sum of |amplitude|^2 held by this rank alone.
    double local_squared_norm() const;

    /// Collect the full state on every rank, ordered by *logical* basis index.
    ///
    /// Only for tests and small debugging runs: the result is 2^n amplitudes
    /// on every rank, which defeats the point of distributing the state.
    std::vector<std::complex<double>> gather() const;

    const LocalMetrics& metrics() const { return metrics_; }
    void reset_metrics() { metrics_ = LocalMetrics{}; }

  private:
    DistributedLayout layout_;
    std::vector<Amplitude> local_;
    LocalMetrics metrics_{};
};

using DistributedStateVector = DistributedStateVectorT<double>;

extern template class DistributedStateVectorT<double>;
extern template class DistributedStateVectorT<float>;

}  // namespace aegisq
