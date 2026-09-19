#include "aegisq/distributed_layout.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace aegisq {

bool is_power_of_two(int value) {
    return value > 0 && (value & (value - 1)) == 0;
}

int log2_exact(int value) {
    if (!is_power_of_two(value)) {
        throw std::invalid_argument("expected a power of two, got " + std::to_string(value));
    }
    int result = 0;
    while ((1 << result) < value) {
        ++result;
    }
    return result;
}

DistributedLayout::DistributedLayout(int num_qubits, int world_size, int rank)
    : DistributedLayout(num_qubits, world_size, rank, std::vector<int>{}) {}

DistributedLayout::DistributedLayout(int num_qubits, int world_size, int rank,
                                     std::vector<int> logical_to_position)
    : num_qubits_(num_qubits),
      world_size_(world_size),
      rank_(rank),
      num_global_(0),
      num_local_(0),
      logical_to_position_(std::move(logical_to_position)) {
    if (!is_power_of_two(world_size)) {
        throw std::invalid_argument(
            "world size must be a power of two for the current partitioning scheme, got " +
            std::to_string(world_size));
    }
    num_global_ = log2_exact(world_size);
    num_local_ = num_qubits - num_global_;

    if (logical_to_position_.empty()) {
        logical_to_position_.resize(static_cast<std::size_t>(std::max(num_qubits, 0)));
        std::iota(logical_to_position_.begin(), logical_to_position_.end(), 0);
    }
    validate();
}

void DistributedLayout::validate() const {
    if (num_qubits_ < 1) {
        throw std::invalid_argument("a distributed state needs at least one qubit");
    }
    if (num_local_ < 1) {
        throw std::invalid_argument(
            "need at least one local qubit: " + std::to_string(num_qubits_) + " qubits over " +
            std::to_string(world_size_) + " ranks leaves none");
    }
    if (rank_ < 0 || rank_ >= world_size_) {
        throw std::invalid_argument("rank out of range for the given world size");
    }
    if (logical_to_position_.size() != static_cast<std::size_t>(num_qubits_)) {
        throw std::invalid_argument("qubit mapping must cover every logical qubit");
    }
    std::vector<int> seen(static_cast<std::size_t>(num_qubits_), 0);
    for (int position : logical_to_position_) {
        if (position < 0 || position >= num_qubits_) {
            throw std::invalid_argument("qubit mapping contains an out-of-range position");
        }
        if (seen[static_cast<std::size_t>(position)]++) {
            throw std::invalid_argument("qubit mapping must be a permutation");
        }
    }
}

int DistributedLayout::position(int logical_qubit) const {
    if (logical_qubit < 0 || logical_qubit >= num_qubits_) {
        throw std::invalid_argument("logical qubit out of range: " + std::to_string(logical_qubit));
    }
    return logical_to_position_[static_cast<std::size_t>(logical_qubit)];
}

int DistributedLayout::global_position(int logical_qubit) const {
    const int pos = position(logical_qubit);
    if (pos < num_local_) {
        throw std::invalid_argument("qubit " + std::to_string(logical_qubit) + " is not global");
    }
    return pos - num_local_;
}

int DistributedLayout::global_bit_for_rank(int logical_qubit, int rank) const {
    return (rank >> global_position(logical_qubit)) & 1;
}

int DistributedLayout::partner_rank_for_global_qubit(int logical_qubit) const {
    return partner_rank_of(logical_qubit, rank_);
}

int DistributedLayout::partner_rank_of(int logical_qubit, int rank) const {
    return rank ^ (1 << global_position(logical_qubit));
}

std::vector<int> DistributedLayout::local_qubits() const {
    std::vector<int> result(static_cast<std::size_t>(num_local_), -1);
    for (int q = 0; q < num_qubits_; ++q) {
        const int pos = logical_to_position_[static_cast<std::size_t>(q)];
        if (pos < num_local_) {
            result[static_cast<std::size_t>(pos)] = q;
        }
    }
    return result;
}

std::vector<int> DistributedLayout::global_qubits() const {
    std::vector<int> result(static_cast<std::size_t>(num_global_), -1);
    for (int q = 0; q < num_qubits_; ++q) {
        const int pos = logical_to_position_[static_cast<std::size_t>(q)];
        if (pos >= num_local_) {
            result[static_cast<std::size_t>(pos - num_local_)] = q;
        }
    }
    return result;
}

bool DistributedLayout::is_identity_mapping() const {
    for (int q = 0; q < num_qubits_; ++q) {
        if (logical_to_position_[static_cast<std::size_t>(q)] != q) {
            return false;
        }
    }
    return true;
}

std::uint64_t DistributedLayout::to_logical_index(std::uint64_t physical_index) const {
    std::uint64_t logical = 0;
    for (int q = 0; q < num_qubits_; ++q) {
        const int pos = logical_to_position_[static_cast<std::size_t>(q)];
        if ((physical_index >> pos) & 1ULL) {
            logical |= (1ULL << q);
        }
    }
    return logical;
}

std::uint64_t DistributedLayout::to_physical_index(std::uint64_t logical_index) const {
    std::uint64_t physical = 0;
    for (int q = 0; q < num_qubits_; ++q) {
        if ((logical_index >> q) & 1ULL) {
            physical |= (1ULL << logical_to_position_[static_cast<std::size_t>(q)]);
        }
    }
    return physical;
}

std::string DistributedLayout::to_string() const {
    std::string out = "DistributedLayout(qubits=" + std::to_string(num_qubits_) +
                      ", ranks=" + std::to_string(world_size_) + ", rank=" + std::to_string(rank_) +
                      ", local=" + std::to_string(num_local_) +
                      ", global=" + std::to_string(num_global_) + ", global_qubits=[";
    const std::vector<int> globals = global_qubits();
    for (std::size_t i = 0; i < globals.size(); ++i) {
        out += std::to_string(globals[i]);
        if (i + 1 < globals.size()) {
            out += ", ";
        }
    }
    out += "])";
    return out;
}

}  // namespace aegisq
