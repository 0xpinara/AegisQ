#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "aegisq/gate.hpp"

namespace aegisq {

/// A validated gate sequence over a fixed-width qubit register.
class Circuit {
  public:
    explicit Circuit(int num_qubits);

    Circuit& add(const Gate& gate);
    Circuit& add(OpCode opcode, int qubit, double param = 0.0);
    Circuit& add(OpCode opcode, int control, int target, bool /*two_qubit_tag*/);

    int num_qubits() const { return num_qubits_; }
    std::size_t size() const { return gates_.size(); }
    const std::vector<Gate>& gates() const { return gates_; }

    /// Longest chain of gates sharing at least one qubit.
    int depth() const;

    std::string to_string() const;

  private:
    void validate(const Gate& gate) const;

    int num_qubits_;
    std::vector<Gate> gates_;
};

}  // namespace aegisq
