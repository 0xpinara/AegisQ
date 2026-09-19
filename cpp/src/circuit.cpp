#include "aegisq/circuit.hpp"

#include <algorithm>
#include <stdexcept>

namespace aegisq {

Circuit::Circuit(int num_qubits) : num_qubits_(num_qubits) {
    if (num_qubits < 1) {
        throw std::invalid_argument("a circuit needs at least one qubit");
    }
    if (num_qubits > 62) {
        throw std::invalid_argument("qubit count must fit a 64-bit basis index");
    }
}

void Circuit::validate(const Gate& gate) const {
    const int arity = gate_arity(gate.opcode);
    for (int i = 0; i < arity; ++i) {
        const int q = gate.qubits[static_cast<std::size_t>(i)];
        if (q < 0 || q >= num_qubits_) {
            throw std::invalid_argument("qubit index out of range in " + aegisq::to_string(gate));
        }
    }
    if (arity == 2 && gate.qubits[0] == gate.qubits[1]) {
        throw std::invalid_argument("two-qubit gate needs distinct operands: " +
                                    aegisq::to_string(gate));
    }
}

Circuit& Circuit::add(const Gate& gate) {
    validate(gate);
    gates_.push_back(gate);
    return *this;
}

Circuit& Circuit::add(OpCode opcode, int qubit, double param) {
    return add(Gate::one(opcode, qubit, param));
}

Circuit& Circuit::add(OpCode opcode, int control, int target, bool) {
    return add(Gate::two(opcode, control, target));
}

int Circuit::depth() const {
    std::vector<int> frontier(static_cast<std::size_t>(num_qubits_), 0);
    for (const Gate& gate : gates_) {
        const int arity = gate_arity(gate.opcode);
        int level = 0;
        for (int i = 0; i < arity; ++i) {
            level = std::max(
                level,
                frontier[static_cast<std::size_t>(gate.qubits[static_cast<std::size_t>(i)])]);
        }
        ++level;
        for (int i = 0; i < arity; ++i) {
            frontier[static_cast<std::size_t>(gate.qubits[static_cast<std::size_t>(i)])] = level;
        }
    }
    return frontier.empty() ? 0 : *std::max_element(frontier.begin(), frontier.end());
}

std::string Circuit::to_string() const {
    std::string out = "Circuit(qubits=" + std::to_string(num_qubits_) +
                      ", gates=" + std::to_string(gates_.size()) + ")";
    for (const Gate& gate : gates_) {
        out += "\n  " + aegisq::to_string(gate);
    }
    return out;
}

}  // namespace aegisq
