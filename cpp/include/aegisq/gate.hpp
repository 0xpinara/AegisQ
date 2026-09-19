#pragma once

#include <array>
#include <complex>
#include <cstdint>
#include <string>
#include <string_view>

namespace aegisq {

/// Supported operations. The set is intentionally small: every opcode must be
/// implemented by the local kernels, the distributed runtime and the
/// communication cost model.
enum class OpCode : std::uint8_t {
    X,
    Y,
    Z,
    H,
    S,
    T,
    RX,
    RY,
    RZ,
    CX,
    CZ,
    SWAP,
};

/// One instruction. `qubits[1]` is unused (-1) for single-qubit operations and
/// `param` is unused (0) for non-parametric ones.
struct Gate {
    OpCode opcode{OpCode::X};
    std::array<int, 2> qubits{-1, -1};
    double param{0.0};

    static Gate one(OpCode op, int qubit, double param = 0.0) {
        return Gate{op, {qubit, -1}, param};
    }
    static Gate two(OpCode op, int a, int b) { return Gate{op, {a, b}, 0.0}; }
};

/// Number of qubit operands the opcode consumes.
int gate_arity(OpCode opcode);

/// True when the matrix is diagonal in the computational basis.
///
/// This is the single most important structural property for the distributed
/// runtime: a diagonal gate scales each amplitude in place and therefore never
/// requires communication, whichever rank holds the amplitude.
bool gate_is_diagonal(OpCode opcode);

/// Index of the control operand, or -1 when the gate has no control.
int gate_control_position(OpCode opcode);

std::string_view opcode_name(OpCode opcode);
OpCode opcode_from_name(std::string_view name);

/// Row-major 2x2 matrix {m00, m01, m10, m11} of a single-qubit gate.
std::array<std::complex<double>, 4> single_qubit_matrix(const Gate& gate);

/// Diagonal entries {d0, d1} of a diagonal single-qubit gate.
std::array<std::complex<double>, 2> diagonal_entries(const Gate& gate);

std::string to_string(const Gate& gate);

}  // namespace aegisq
