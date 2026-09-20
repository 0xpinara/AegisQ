#include "aegisq/statevector.hpp"

#include <chrono>
#include <stdexcept>

#include "aegisq/kernels.hpp"

namespace aegisq {
namespace {

/// Guard against an accidental 2^40 allocation on a workstation.
constexpr int kMaxSingleProcessQubits = 34;

class ScopedTimer {
  public:
    explicit ScopedTimer(double& sink) : sink_(sink), start_(std::chrono::steady_clock::now()) {}
    ~ScopedTimer() {
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start_;
        sink_ += elapsed.count();
    }

  private:
    double& sink_;
    std::chrono::steady_clock::time_point start_;
};

}  // namespace

template <typename Real>
StateVectorT<Real>::StateVectorT(int num_qubits) : num_qubits_(num_qubits) {
    if (num_qubits < 1) {
        throw std::invalid_argument("state vector needs at least one qubit");
    }
    if (num_qubits > kMaxSingleProcessQubits) {
        throw std::invalid_argument(
            "single-process state vector limited to 34 qubits; use the distributed runtime");
    }
    amplitudes_.assign(std::size_t{1} << num_qubits, Amplitude{0, 0});
    amplitudes_[0] = Amplitude{1, 0};
}

template <typename Real>
void StateVectorT<Real>::reset() {
    std::fill(amplitudes_.begin(), amplitudes_.end(), Amplitude{0, 0});
    amplitudes_[0] = Amplitude{1, 0};
    metrics_ = LocalMetrics{};
}

template <typename Real>
void StateVectorT<Real>::set_amplitudes(const std::vector<Amplitude>& values) {
    if (values.size() != amplitudes_.size()) {
        throw std::invalid_argument("amplitude vector has the wrong length");
    }
    amplitudes_ = values;
}

template <typename Real>
void StateVectorT<Real>::check_qubit(int qubit) const {
    if (qubit < 0 || qubit >= num_qubits_) {
        throw std::invalid_argument("qubit index out of range");
    }
}

template <typename Real>
double StateVectorT<Real>::norm() const {
    return kernels::squared_norm(amplitudes_.data(), amplitudes_.size());
}

template <typename Real>
std::vector<double> StateVectorT<Real>::probabilities() const {
    std::vector<double> probs(amplitudes_.size());
    for (std::size_t i = 0; i < amplitudes_.size(); ++i) {
        const double re = static_cast<double>(amplitudes_[i].real());
        const double im = static_cast<double>(amplitudes_[i].imag());
        probs[i] = re * re + im * im;
    }
    return probs;
}

// Named helpers forward to apply_gate so validation and timing live in one place.
template <typename Real>
void StateVectorT<Real>::apply_x(int qubit) {
    apply_gate(Gate::one(OpCode::X, qubit));
}
template <typename Real>
void StateVectorT<Real>::apply_y(int qubit) {
    apply_gate(Gate::one(OpCode::Y, qubit));
}
template <typename Real>
void StateVectorT<Real>::apply_z(int qubit) {
    apply_gate(Gate::one(OpCode::Z, qubit));
}
template <typename Real>
void StateVectorT<Real>::apply_h(int qubit) {
    apply_gate(Gate::one(OpCode::H, qubit));
}
template <typename Real>
void StateVectorT<Real>::apply_s(int qubit) {
    apply_gate(Gate::one(OpCode::S, qubit));
}
template <typename Real>
void StateVectorT<Real>::apply_t(int qubit) {
    apply_gate(Gate::one(OpCode::T, qubit));
}
template <typename Real>
void StateVectorT<Real>::apply_rx(int qubit, double theta) {
    apply_gate(Gate::one(OpCode::RX, qubit, theta));
}
template <typename Real>
void StateVectorT<Real>::apply_ry(int qubit, double theta) {
    apply_gate(Gate::one(OpCode::RY, qubit, theta));
}
template <typename Real>
void StateVectorT<Real>::apply_rz(int qubit, double theta) {
    apply_gate(Gate::one(OpCode::RZ, qubit, theta));
}
template <typename Real>
void StateVectorT<Real>::apply_cnot(int control, int target) {
    apply_gate(Gate::two(OpCode::CX, control, target));
}
template <typename Real>
void StateVectorT<Real>::apply_cz(int a, int b) {
    apply_gate(Gate::two(OpCode::CZ, a, b));
}
template <typename Real>
void StateVectorT<Real>::apply_swap(int a, int b) {
    apply_gate(Gate::two(OpCode::SWAP, a, b));
}

template <typename Real>
void StateVectorT<Real>::apply_gate(const Gate& gate) {
    const int arity = gate_arity(gate.opcode);
    for (int i = 0; i < arity; ++i) {
        check_qubit(gate.qubits[static_cast<std::size_t>(i)]);
    }
    if (arity == 2 && gate.qubits[0] == gate.qubits[1]) {
        throw std::invalid_argument("two-qubit gate needs distinct operands");
    }

    ScopedTimer timer(metrics_.compute_seconds);
    Amplitude* psi = amplitudes_.data();
    const std::size_t n = amplitudes_.size();

    switch (gate.opcode) {
        case OpCode::X:
        case OpCode::Y:
        case OpCode::H:
        case OpCode::RX:
        case OpCode::RY:
            kernels::apply_single_qubit(psi, n, gate.qubits[0], single_qubit_matrix(gate));
            break;
        case OpCode::Z:
        case OpCode::S:
        case OpCode::T:
        case OpCode::RZ:
            kernels::apply_diagonal(psi, n, gate.qubits[0], diagonal_entries(gate));
            break;
        case OpCode::CX:
            kernels::apply_controlled_single_qubit(psi, n, gate.qubits[0], gate.qubits[1],
                                                   single_qubit_matrix(Gate::one(OpCode::X, 0)));
            break;
        case OpCode::CZ:
            kernels::apply_cz(psi, n, gate.qubits[0], gate.qubits[1]);
            break;
        case OpCode::SWAP:
            kernels::apply_swap(psi, n, gate.qubits[0], gate.qubits[1]);
            break;
        case OpCode::U:
            // A fused gate takes the cheaper diagonal kernel whenever its
            // matrix allows it, exactly as the primitive gates do.
            if (gate_is_diagonal(gate)) {
                kernels::apply_diagonal(psi, n, gate.qubits[0], diagonal_entries(gate));
            } else {
                kernels::apply_single_qubit(psi, n, gate.qubits[0], gate.matrix);
            }
            break;
    }
    ++metrics_.gates_applied;
}

template <typename Real>
void StateVectorT<Real>::apply_circuit(const Circuit& circuit) {
    if (circuit.num_qubits() != num_qubits_) {
        throw std::invalid_argument("circuit width does not match the state vector");
    }
    for (const Gate& gate : circuit.gates()) {
        apply_gate(gate);
    }
}

template <typename Real>
MeasurementResult StateVectorT<Real>::measure_all(std::uint64_t shots, std::uint64_t seed) const {
    MeasurementResult result;
    result.shots = shots;
    result.seed = seed;
    if (shots == 0) {
        return result;
    }
    const double total = norm();
    const std::vector<double> draws = sorted_uniform_draws(shots, seed, total);
    accumulate_shots(amplitudes_.data(), amplitudes_.size(), 0, 0.0, draws, result.counts);
    return result;
}

template class StateVectorT<double>;
template class StateVectorT<float>;

}  // namespace aegisq
