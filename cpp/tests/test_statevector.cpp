#include <cmath>
#include <complex>

#include "aegisq/statevector.hpp"
#include "testing.hpp"

using aegisq::Circuit;
using aegisq::Gate;
using aegisq::OpCode;
using aegisq::StateVector;

namespace {

constexpr double kTol = 1e-12;
const double kInvSqrt2 = 1.0 / std::sqrt(2.0);

}  // namespace

int main() {
    {  // |0...0> initial state
        StateVector sv(3);
        AEGISQ_CHECK_CLOSE(sv.amplitudes()[0], std::complex<double>(1, 0), kTol);
        AEGISQ_CHECK_CLOSE(sv.norm(), 1.0, kTol);
    }

    {  // X on qubit 1 moves |000> to basis index 2
        StateVector sv(3);
        sv.apply_x(1);
        AEGISQ_CHECK_CLOSE(sv.amplitudes()[2], std::complex<double>(1, 0), kTol);
    }

    {  // Bell state
        StateVector sv(2);
        sv.apply_h(0);
        sv.apply_cnot(0, 1);
        AEGISQ_CHECK_CLOSE(sv.amplitudes()[0], std::complex<double>(kInvSqrt2, 0), kTol);
        AEGISQ_CHECK_CLOSE(sv.amplitudes()[3], std::complex<double>(kInvSqrt2, 0), kTol);
        AEGISQ_CHECK_CLOSE(sv.norm(), 1.0, kTol);
    }

    {  // GHZ over 10 qubits keeps the norm
        StateVector sv(10);
        sv.apply_h(0);
        for (int q = 0; q < 9; ++q) {
            sv.apply_cnot(q, q + 1);
        }
        AEGISQ_CHECK_CLOSE(sv.norm(), 1.0, kTol);
        AEGISQ_CHECK_CLOSE(sv.amplitudes()[0], std::complex<double>(kInvSqrt2, 0), kTol);
        AEGISQ_CHECK_CLOSE(sv.amplitudes()[(1u << 10) - 1], std::complex<double>(kInvSqrt2, 0),
                           kTol);
    }

    {  // Involutions and rotation inverses restore the state exactly
        StateVector sv(6);
        sv.apply_h(0);
        sv.apply_cnot(0, 3);
        sv.apply_ry(4, 0.9);
        const auto reference = sv.amplitudes();

        sv.apply_x(2);
        sv.apply_x(2);
        sv.apply_h(5);
        sv.apply_h(5);
        sv.apply_rz(1, 0.7);
        sv.apply_rz(1, -0.7);
        sv.apply_rx(3, -1.3);
        sv.apply_rx(3, 1.3);
        sv.apply_swap(0, 2);
        sv.apply_swap(0, 2);

        for (std::size_t i = 0; i < reference.size(); ++i) {
            AEGISQ_CHECK_CLOSE(sv.amplitudes()[i], reference[i], 1e-12);
        }
    }

    {  // CZ only phases the |11> component
        StateVector sv(2);
        sv.apply_h(0);
        sv.apply_h(1);
        sv.apply_cz(0, 1);
        AEGISQ_CHECK_CLOSE(sv.amplitudes()[3], std::complex<double>(-0.5, 0), kTol);
        AEGISQ_CHECK_CLOSE(sv.amplitudes()[1], std::complex<double>(0.5, 0), kTol);
    }

    {  // Circuit application matches the same gates applied one by one
        Circuit circuit(4);
        circuit.add(Gate::one(OpCode::H, 0));
        circuit.add(Gate::two(OpCode::CX, 0, 1));
        circuit.add(Gate::one(OpCode::RY, 2, 0.5));
        circuit.add(Gate::two(OpCode::CZ, 1, 3));

        StateVector a(4);
        a.apply_circuit(circuit);

        StateVector b(4);
        b.apply_h(0);
        b.apply_cnot(0, 1);
        b.apply_ry(2, 0.5);
        b.apply_cz(1, 3);

        for (std::size_t i = 0; i < a.size(); ++i) {
            AEGISQ_CHECK_CLOSE(a.amplitudes()[i], b.amplitudes()[i], kTol);
        }
        AEGISQ_CHECK(a.metrics().gates_applied == 4);
        AEGISQ_CHECK(circuit.depth() == 3);
    }

    {  // Measurement is deterministic for a fixed seed and respects the state
        StateVector sv(3);
        sv.apply_h(0);
        sv.apply_cnot(0, 1);
        sv.apply_cnot(1, 2);
        const auto first = sv.measure_all(512, 7);
        const auto second = sv.measure_all(512, 7);
        AEGISQ_CHECK(first.counts == second.counts);

        std::uint64_t total = 0;
        for (const auto& [index, count] : first.counts) {
            AEGISQ_CHECK(index == 0 || index == 7);  // GHZ support
            total += count;
        }
        AEGISQ_CHECK(total == 512);

        const auto other = sv.measure_all(512, 8);
        AEGISQ_CHECK(other.counts != first.counts);
    }

    {  // fp32 tracks fp64 within single-precision tolerance
        aegisq::StateVectorF32 small(5);
        StateVector large(5);
        for (int q = 0; q < 5; ++q) {
            small.apply_h(q);
            large.apply_h(q);
        }
        small.apply_cnot(0, 4);
        large.apply_cnot(0, 4);
        for (std::size_t i = 0; i < large.size(); ++i) {
            AEGISQ_CHECK_CLOSE(static_cast<std::complex<double>>(small.amplitudes()[i]),
                               large.amplitudes()[i], 1e-6);
        }
    }

    {  // Invalid operands are rejected
        StateVector sv(2);
        bool threw = false;
        try {
            sv.apply_h(5);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        AEGISQ_CHECK(threw);

        threw = false;
        try {
            sv.apply_cnot(1, 1);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        AEGISQ_CHECK(threw);
    }

    return aegisq::testing::summary("test_statevector");
}
