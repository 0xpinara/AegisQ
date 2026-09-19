#include <array>
#include <complex>

#include "aegisq/gate.hpp"
#include "testing.hpp"

using aegisq::Gate;
using aegisq::OpCode;

namespace {

bool is_unitary(const std::array<std::complex<double>, 4>& m) {
    // U U^dagger == I
    const std::complex<double> a = m[0] * std::conj(m[0]) + m[1] * std::conj(m[1]);
    const std::complex<double> b = m[0] * std::conj(m[2]) + m[1] * std::conj(m[3]);
    const std::complex<double> d = m[2] * std::conj(m[2]) + m[3] * std::conj(m[3]);
    return std::abs(a - 1.0) < 1e-12 && std::abs(b) < 1e-12 && std::abs(d - 1.0) < 1e-12;
}

}  // namespace

int main() {
    const std::array<OpCode, 6> static_ops{OpCode::X, OpCode::Y, OpCode::Z,
                                           OpCode::H, OpCode::S, OpCode::T};
    for (OpCode op : static_ops) {
        AEGISQ_CHECK(is_unitary(aegisq::single_qubit_matrix(Gate::one(op, 0))));
    }

    for (double theta : {0.0, 0.3, -1.7, 3.14159}) {
        AEGISQ_CHECK(is_unitary(aegisq::single_qubit_matrix(Gate::one(OpCode::RX, 0, theta))));
        AEGISQ_CHECK(is_unitary(aegisq::single_qubit_matrix(Gate::one(OpCode::RY, 0, theta))));
        AEGISQ_CHECK(is_unitary(aegisq::single_qubit_matrix(Gate::one(OpCode::RZ, 0, theta))));
    }

    // The diagonal classification drives every communication decision later on.
    AEGISQ_CHECK(aegisq::gate_is_diagonal(OpCode::Z));
    AEGISQ_CHECK(aegisq::gate_is_diagonal(OpCode::S));
    AEGISQ_CHECK(aegisq::gate_is_diagonal(OpCode::T));
    AEGISQ_CHECK(aegisq::gate_is_diagonal(OpCode::RZ));
    AEGISQ_CHECK(aegisq::gate_is_diagonal(OpCode::CZ));
    AEGISQ_CHECK(!aegisq::gate_is_diagonal(OpCode::X));
    AEGISQ_CHECK(!aegisq::gate_is_diagonal(OpCode::H));
    AEGISQ_CHECK(!aegisq::gate_is_diagonal(OpCode::RX));
    AEGISQ_CHECK(!aegisq::gate_is_diagonal(OpCode::CX));
    AEGISQ_CHECK(!aegisq::gate_is_diagonal(OpCode::SWAP));

    AEGISQ_CHECK(aegisq::gate_arity(OpCode::H) == 1);
    AEGISQ_CHECK(aegisq::gate_arity(OpCode::CX) == 2);
    AEGISQ_CHECK(aegisq::gate_control_position(OpCode::CX) == 0);
    AEGISQ_CHECK(aegisq::gate_control_position(OpCode::H) == -1);

    AEGISQ_CHECK(aegisq::opcode_from_name("swap") == OpCode::SWAP);
    AEGISQ_CHECK(aegisq::opcode_name(OpCode::RY) == "ry");

    bool threw = false;
    try {
        aegisq::opcode_from_name("toffoli");
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    AEGISQ_CHECK(threw);

    return aegisq::testing::summary("test_gates");
}
