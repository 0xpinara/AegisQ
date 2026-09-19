#pragma once

/// How a state vector is spread over MPI ranks.
///
/// With `P = 2^p` ranks and `n` qubits, each rank owns `2^L` amplitudes where
/// `L = n - p`. A *physical* basis index is split as
///
///     physical = (rank << L) | local_index
///
/// so the low `L` bit positions are **local** (addressable inside a rank's
/// shard) and the high `p` bit positions are **global** (they select the rank).
///
/// Which logical qubit occupies which physical position is a free choice, and
/// it is the choice this whole project is about: a logical qubit parked on a
/// global position turns some of its gates into network operations. The layout
/// therefore carries an explicit permutation `logical_to_position`, defaulting
/// to the identity (qubit q at position q, so the highest-numbered qubits are
/// the global ones). The communication-aware mapper replaces that permutation.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace aegisq {

bool is_power_of_two(int value);

/// Exact base-2 logarithm of a power of two.
int log2_exact(int value);

class DistributedLayout {
  public:
    /// Identity mapping: logical qubit q sits at physical position q.
    DistributedLayout(int num_qubits, int world_size, int rank);

    /// Explicit mapping; `logical_to_position` must be a permutation of [0, n).
    DistributedLayout(int num_qubits, int world_size, int rank,
                      std::vector<int> logical_to_position);

    int num_qubits() const { return num_qubits_; }
    int world_size() const { return world_size_; }
    int rank() const { return rank_; }

    /// Number of global (rank-selecting) qubits, p = log2(world_size).
    int num_global_qubits() const { return num_global_; }

    /// Number of local qubits, L = n - p.
    int num_local_qubits() const { return num_local_; }

    /// Amplitudes stored per rank, 2^L.
    std::size_t local_state_size() const { return std::size_t{1} << num_local_; }

    /// Bytes per rank for a given amplitude size.
    std::size_t local_state_bytes(std::size_t amplitude_bytes) const {
        return local_state_size() * amplitude_bytes;
    }

    /// Physical bit position of a logical qubit.
    int position(int logical_qubit) const;

    bool is_local(int logical_qubit) const { return position(logical_qubit) < num_local_; }
    bool is_global(int logical_qubit) const { return position(logical_qubit) >= num_local_; }

    /// Index of a global qubit among the global positions, in [0, p).
    int global_position(int logical_qubit) const;

    /// Value (0 or 1) that a global qubit takes on a given rank.
    int global_bit_for_rank(int logical_qubit, int rank) const;
    int global_bit(int logical_qubit) const { return global_bit_for_rank(logical_qubit, rank_); }

    /// The rank holding the amplitudes that differ only in this global qubit.
    int partner_rank_for_global_qubit(int logical_qubit) const;
    int partner_rank_of(int logical_qubit, int rank) const;

    /// Logical qubits sitting on local positions, ordered by position.
    std::vector<int> local_qubits() const;

    /// Logical qubits sitting on global positions, ordered by position.
    std::vector<int> global_qubits() const;

    const std::vector<int>& logical_to_position() const { return logical_to_position_; }
    bool is_identity_mapping() const;

    /// Physical basis index of a local element on this rank.
    std::uint64_t physical_index(std::uint64_t local_index) const {
        return (static_cast<std::uint64_t>(rank_) << num_local_) | local_index;
    }
    std::uint64_t physical_index_for_rank(int rank, std::uint64_t local_index) const {
        return (static_cast<std::uint64_t>(rank) << num_local_) | local_index;
    }

    /// Translate between physical (position-ordered) and logical basis indices.
    std::uint64_t to_logical_index(std::uint64_t physical_index) const;
    std::uint64_t to_physical_index(std::uint64_t logical_index) const;

    std::string to_string() const;

  private:
    void validate() const;

    int num_qubits_;
    int world_size_;
    int rank_;
    int num_global_;
    int num_local_;
    std::vector<int> logical_to_position_;
};

}  // namespace aegisq
