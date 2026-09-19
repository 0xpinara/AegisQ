#pragma once

/// Local state-vector kernels.
///
/// Every kernel works on a contiguous block of amplitudes addressed by *local*
/// qubit indices. That is deliberate: the distributed runtime hands each rank
/// its own shard and reuses exactly these kernels for every gate that does not
/// need communication, so single-process and distributed execution share one
/// arithmetic implementation and cannot drift apart.
///
/// Index arithmetic follows the paired-index pattern. For a gate on qubit `q`
/// the amplitudes group into pairs `(i, i | 2^q)` that differ only in bit `q`;
/// enumerating the `2^(n-1)` pair representatives touches each amplitude once.

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>

namespace aegisq::kernels {

/// Below this many pairs OpenMP's fork/join costs more than the work itself.
inline constexpr std::size_t kParallelThreshold = 1ULL << 12;

using Matrix2 = std::array<std::complex<double>, 4>;
using Diagonal2 = std::array<std::complex<double>, 2>;

namespace detail {

/// Insert a zero bit at position `pos` into `value`.
inline std::size_t insert_zero_bit(std::size_t value, int pos) {
    const std::size_t mask = (std::size_t{1} << pos) - 1;
    return ((value & ~mask) << 1) | (value & mask);
}

/// Expand a compact counter into an index with zeros at both `a` and `b`.
inline std::size_t insert_two_zero_bits(std::size_t value, int a, int b) {
    const int lo = a < b ? a : b;
    const int hi = a < b ? b : a;
    return insert_zero_bit(insert_zero_bit(value, lo), hi);
}

}  // namespace detail

/// General 2x2 unitary on a local qubit.
template <typename Amp>
void apply_single_qubit(Amp* psi, std::size_t size, int qubit, const Matrix2& m) {
    using Real = typename Amp::value_type;
    const auto m00 = Amp(static_cast<Real>(m[0].real()), static_cast<Real>(m[0].imag()));
    const auto m01 = Amp(static_cast<Real>(m[1].real()), static_cast<Real>(m[1].imag()));
    const auto m10 = Amp(static_cast<Real>(m[2].real()), static_cast<Real>(m[2].imag()));
    const auto m11 = Amp(static_cast<Real>(m[3].real()), static_cast<Real>(m[3].imag()));

    const std::size_t stride = std::size_t{1} << qubit;
    const std::size_t pairs = size >> 1;
    const std::size_t low_mask = stride - 1;

#pragma omp parallel for schedule(static) if (pairs >= kParallelThreshold)
    for (std::ptrdiff_t k = 0; k < static_cast<std::ptrdiff_t>(pairs); ++k) {
        const std::size_t kk = static_cast<std::size_t>(k);
        const std::size_t i0 = ((kk & ~low_mask) << 1) | (kk & low_mask);
        const std::size_t i1 = i0 | stride;
        const Amp a = psi[i0];
        const Amp b = psi[i1];
        psi[i0] = m00 * a + m01 * b;
        psi[i1] = m10 * a + m11 * b;
    }
}

/// Diagonal gate on a local qubit: scale by `d[0]` or `d[1]` per bit value.
template <typename Amp>
void apply_diagonal(Amp* psi, std::size_t size, int qubit, const Diagonal2& d) {
    using Real = typename Amp::value_type;
    const auto d0 = Amp(static_cast<Real>(d[0].real()), static_cast<Real>(d[0].imag()));
    const auto d1 = Amp(static_cast<Real>(d[1].real()), static_cast<Real>(d[1].imag()));

    const std::size_t stride = std::size_t{1} << qubit;
    const std::size_t pairs = size >> 1;
    const std::size_t low_mask = stride - 1;

#pragma omp parallel for schedule(static) if (pairs >= kParallelThreshold)
    for (std::ptrdiff_t k = 0; k < static_cast<std::ptrdiff_t>(pairs); ++k) {
        const std::size_t kk = static_cast<std::size_t>(k);
        const std::size_t i0 = ((kk & ~low_mask) << 1) | (kk & low_mask);
        psi[i0] *= d0;
        psi[i0 | stride] *= d1;
    }
}

/// Multiply the whole shard by one scalar.
///
/// Used when a diagonal gate acts on a *global* qubit: every amplitude on the
/// rank shares the same bit value, so the gate collapses to a single scale.
template <typename Amp>
void scale_all(Amp* psi, std::size_t size, std::complex<double> factor) {
    using Real = typename Amp::value_type;
    const auto f = Amp(static_cast<Real>(factor.real()), static_cast<Real>(factor.imag()));
#pragma omp parallel for schedule(static) if (size >= kParallelThreshold)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(size); ++i) {
        psi[static_cast<std::size_t>(i)] *= f;
    }
}

/// 2x2 unitary on `target`, applied only where local `control` bit is one.
template <typename Amp>
void apply_controlled_single_qubit(Amp* psi, std::size_t size, int control, int target,
                                   const Matrix2& m) {
    using Real = typename Amp::value_type;
    const auto m00 = Amp(static_cast<Real>(m[0].real()), static_cast<Real>(m[0].imag()));
    const auto m01 = Amp(static_cast<Real>(m[1].real()), static_cast<Real>(m[1].imag()));
    const auto m10 = Amp(static_cast<Real>(m[2].real()), static_cast<Real>(m[2].imag()));
    const auto m11 = Amp(static_cast<Real>(m[3].real()), static_cast<Real>(m[3].imag()));

    const std::size_t control_bit = std::size_t{1} << control;
    const std::size_t target_bit = std::size_t{1} << target;
    const std::size_t quads = size >> 2;

#pragma omp parallel for schedule(static) if (quads >= kParallelThreshold)
    for (std::ptrdiff_t k = 0; k < static_cast<std::ptrdiff_t>(quads); ++k) {
        const std::size_t base =
            detail::insert_two_zero_bits(static_cast<std::size_t>(k), control, target);
        const std::size_t i0 = base | control_bit;
        const std::size_t i1 = i0 | target_bit;
        const Amp a = psi[i0];
        const Amp b = psi[i1];
        psi[i0] = m00 * a + m01 * b;
        psi[i1] = m10 * a + m11 * b;
    }
}

/// Controlled-Z between two local qubits: negate the |11> component.
template <typename Amp>
void apply_cz(Amp* psi, std::size_t size, int a, int b) {
    const std::size_t bit_a = std::size_t{1} << a;
    const std::size_t bit_b = std::size_t{1} << b;
    const std::size_t quads = size >> 2;

#pragma omp parallel for schedule(static) if (quads >= kParallelThreshold)
    for (std::ptrdiff_t k = 0; k < static_cast<std::ptrdiff_t>(quads); ++k) {
        const std::size_t index =
            detail::insert_two_zero_bits(static_cast<std::size_t>(k), a, b) | bit_a | bit_b;
        psi[index] = -psi[index];
    }
}

/// Apply a phase to amplitudes whose local `qubit` bit is one.
///
/// This is the shard-local half of a controlled diagonal gate whose control
/// lives on a global qubit: the rank already knows the control is satisfied.
template <typename Amp>
void apply_phase_if_bit_set(Amp* psi, std::size_t size, int qubit, std::complex<double> phase) {
    using Real = typename Amp::value_type;
    const auto p = Amp(static_cast<Real>(phase.real()), static_cast<Real>(phase.imag()));
    const std::size_t stride = std::size_t{1} << qubit;
    const std::size_t pairs = size >> 1;
    const std::size_t low_mask = stride - 1;

#pragma omp parallel for schedule(static) if (pairs >= kParallelThreshold)
    for (std::ptrdiff_t k = 0; k < static_cast<std::ptrdiff_t>(pairs); ++k) {
        const std::size_t kk = static_cast<std::size_t>(k);
        const std::size_t i0 = ((kk & ~low_mask) << 1) | (kk & low_mask);
        psi[i0 | stride] *= p;
    }
}

/// SWAP between two local qubits.
template <typename Amp>
void apply_swap(Amp* psi, std::size_t size, int a, int b) {
    const std::size_t bit_a = std::size_t{1} << a;
    const std::size_t bit_b = std::size_t{1} << b;
    const std::size_t quads = size >> 2;

#pragma omp parallel for schedule(static) if (quads >= kParallelThreshold)
    for (std::ptrdiff_t k = 0; k < static_cast<std::ptrdiff_t>(quads); ++k) {
        const std::size_t base = detail::insert_two_zero_bits(static_cast<std::size_t>(k), a, b);
        const std::size_t i01 = base | bit_b;
        const std::size_t i10 = base | bit_a;
        const Amp tmp = psi[i01];
        psi[i01] = psi[i10];
        psi[i10] = tmp;
    }
}

/// Apply a 2x2 unitary whose two basis components live on different ranks.
///
/// `local` holds this rank's amplitudes and `remote` the partner's, both
/// indexed by the same local index. `local_bit` is the value the gate's qubit
/// takes on this rank (0 or 1), which selects the row of the matrix:
///
///     bit 0:  local[i] <- m00 * local[i] + m01 * remote[i]
///     bit 1:  local[i] <- m10 * remote[i] + m11 * local[i]
///
/// The partner rank runs the same routine with the opposite bit, so between
/// them they compute both rows. `remote` is a separate buffer, so no source
/// value is overwritten before it is consumed.
template <typename Amp>
void apply_single_qubit_paired(Amp* local, const Amp* remote, std::size_t size, const Matrix2& m,
                               int local_bit) {
    using Real = typename Amp::value_type;
    const std::size_t self_index = local_bit == 0 ? 0 : 3;
    const std::size_t other_index = local_bit == 0 ? 1 : 2;
    const auto self_coeff =
        Amp(static_cast<Real>(m[self_index].real()), static_cast<Real>(m[self_index].imag()));
    const auto other_coeff =
        Amp(static_cast<Real>(m[other_index].real()), static_cast<Real>(m[other_index].imag()));

#pragma omp parallel for schedule(static) if (size >= kParallelThreshold)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(size); ++i) {
        const std::size_t index = static_cast<std::size_t>(i);
        local[index] = self_coeff * local[index] + other_coeff * remote[index];
    }
}

/// Overwrite a contiguous run of local amplitudes with the partner's values.
///
/// Used where a gate permutes basis states across ranks (a global `X`, or the
/// target half of a `CX`): the arithmetic is a copy, the cost is the transfer.
template <typename Amp>
void overwrite_from(Amp* local, const Amp* remote, std::size_t size) {
#pragma omp parallel for schedule(static) if (size >= kParallelThreshold)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(size); ++i) {
        local[static_cast<std::size_t>(i)] = remote[static_cast<std::size_t>(i)];
    }
}

/// Sum of |amplitude|^2 over the block.
template <typename Amp>
double squared_norm(const Amp* psi, std::size_t size) {
    double total = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : total) if (size >= kParallelThreshold)
    for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(size); ++i) {
        const Amp value = psi[static_cast<std::size_t>(i)];
        const double re = static_cast<double>(value.real());
        const double im = static_cast<double>(value.imag());
        total += re * re + im * im;
    }
    return total;
}

}  // namespace aegisq::kernels
