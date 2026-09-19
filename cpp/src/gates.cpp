#include "aegisq/gate.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>

namespace aegisq {
namespace {

constexpr double kInvSqrt2 = 0.7071067811865475244;

}  // namespace

int gate_arity(OpCode opcode) {
    switch (opcode) {
        case OpCode::CX:
        case OpCode::CZ:
        case OpCode::SWAP:
            return 2;
        default:
            return 1;
    }
}

bool gate_is_diagonal(OpCode opcode) {
    switch (opcode) {
        case OpCode::Z:
        case OpCode::S:
        case OpCode::T:
        case OpCode::RZ:
        case OpCode::CZ:
            return true;
        default:
            return false;
    }
}

int gate_control_position(OpCode opcode) {
    switch (opcode) {
        case OpCode::CX:
        case OpCode::CZ:
            return 0;
        default:
            return -1;
    }
}

std::string_view opcode_name(OpCode opcode) {
    switch (opcode) {
        case OpCode::X:
            return "x";
        case OpCode::Y:
            return "y";
        case OpCode::Z:
            return "z";
        case OpCode::H:
            return "h";
        case OpCode::S:
            return "s";
        case OpCode::T:
            return "t";
        case OpCode::RX:
            return "rx";
        case OpCode::RY:
            return "ry";
        case OpCode::RZ:
            return "rz";
        case OpCode::CX:
            return "cx";
        case OpCode::CZ:
            return "cz";
        case OpCode::SWAP:
            return "swap";
    }
    return "unknown";
}

OpCode opcode_from_name(std::string_view name) {
    if (name == "x") return OpCode::X;
    if (name == "y") return OpCode::Y;
    if (name == "z") return OpCode::Z;
    if (name == "h") return OpCode::H;
    if (name == "s") return OpCode::S;
    if (name == "t") return OpCode::T;
    if (name == "rx") return OpCode::RX;
    if (name == "ry") return OpCode::RY;
    if (name == "rz") return OpCode::RZ;
    if (name == "cx") return OpCode::CX;
    if (name == "cz") return OpCode::CZ;
    if (name == "swap") return OpCode::SWAP;
    throw std::invalid_argument("unsupported opcode: " + std::string(name));
}

std::array<std::complex<double>, 4> single_qubit_matrix(const Gate& gate) {
    using C = std::complex<double>;
    const double theta = gate.param;
    switch (gate.opcode) {
        case OpCode::X:
            return {C{0, 0}, C{1, 0}, C{1, 0}, C{0, 0}};
        case OpCode::Y:
            return {C{0, 0}, C{0, -1}, C{0, 1}, C{0, 0}};
        case OpCode::Z:
            return {C{1, 0}, C{0, 0}, C{0, 0}, C{-1, 0}};
        case OpCode::H:
            return {C{kInvSqrt2, 0}, C{kInvSqrt2, 0}, C{kInvSqrt2, 0}, C{-kInvSqrt2, 0}};
        case OpCode::S:
            return {C{1, 0}, C{0, 0}, C{0, 0}, C{0, 1}};
        case OpCode::T:
            return {C{1, 0}, C{0, 0}, C{0, 0}, C{kInvSqrt2, kInvSqrt2}};
        case OpCode::RX: {
            const double c = std::cos(theta / 2.0);
            const double s = std::sin(theta / 2.0);
            return {C{c, 0}, C{0, -s}, C{0, -s}, C{c, 0}};
        }
        case OpCode::RY: {
            const double c = std::cos(theta / 2.0);
            const double s = std::sin(theta / 2.0);
            return {C{c, 0}, C{-s, 0}, C{s, 0}, C{c, 0}};
        }
        case OpCode::RZ: {
            const C phase{std::cos(theta / 2.0), -std::sin(theta / 2.0)};
            return {phase, C{0, 0}, C{0, 0}, std::conj(phase)};
        }
        default:
            throw std::invalid_argument("not a single-qubit gate: " +
                                        std::string(opcode_name(gate.opcode)));
    }
}

std::array<std::complex<double>, 2> diagonal_entries(const Gate& gate) {
    if (!gate_is_diagonal(gate.opcode) || gate_arity(gate.opcode) != 1) {
        throw std::invalid_argument("not a diagonal single-qubit gate: " +
                                    std::string(opcode_name(gate.opcode)));
    }
    const auto m = single_qubit_matrix(gate);
    return {m[0], m[3]};
}

std::string to_string(const Gate& gate) {
    std::string out(opcode_name(gate.opcode));
    if (gate.opcode == OpCode::RX || gate.opcode == OpCode::RY || gate.opcode == OpCode::RZ) {
        out += "(" + std::to_string(gate.param) + ")";
    }
    out += " q" + std::to_string(gate.qubits[0]);
    if (gate_arity(gate.opcode) == 2) {
        out += ", q" + std::to_string(gate.qubits[1]);
    }
    return out;
}

}  // namespace aegisq
