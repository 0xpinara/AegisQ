#pragma once

/// Shot sampling that is independent of how the state vector is partitioned.
///
/// The sampler draws `shots` uniform values in `[0, total_probability)` from a
/// seeded Mersenne Twister, sorts them once, and then walks the local
/// amplitudes accumulating probability mass. A rank only claims the draws that
/// fall inside its own probability interval `[offset, offset + local_mass)`.
///
/// Because the draws depend only on `(shots, seed)` and never on the rank
/// count, running the same circuit on 1, 2 or 8 ranks produces *identical*
/// counts. That property is what makes distributed results reproducible and
/// comparable in the benchmark suite.

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <vector>

namespace aegisq {

struct MeasurementResult {
    /// Basis state index -> number of shots that landed on it.
    std::map<std::uint64_t, std::uint64_t> counts;
    std::uint64_t shots{0};
    std::uint64_t seed{0};
};

/// `shots` sorted uniform draws in `[0, total)` from a seeded MT19937-64.
std::vector<double> sorted_uniform_draws(std::uint64_t shots, std::uint64_t seed, double total);

/// Assign the draws that fall into this block's probability interval.
///
/// `index_offset` is the global basis index of local element 0.
/// `probability_offset` is the probability mass held by all preceding blocks.
template <typename Amp>
void accumulate_shots(const Amp* psi, std::size_t size, std::uint64_t index_offset,
                      double probability_offset, const std::vector<double>& sorted_draws,
                      std::map<std::uint64_t, std::uint64_t>& counts) {
    if (sorted_draws.empty() || size == 0) {
        return;
    }
    // Skip draws claimed by earlier blocks.
    auto cursor = std::lower_bound(sorted_draws.begin(), sorted_draws.end(), probability_offset);
    double accumulated = probability_offset;

    for (std::size_t i = 0; i < size && cursor != sorted_draws.end(); ++i) {
        const double re = static_cast<double>(psi[i].real());
        const double im = static_cast<double>(psi[i].imag());
        accumulated += re * re + im * im;
        std::uint64_t hits = 0;
        while (cursor != sorted_draws.end() && *cursor < accumulated) {
            ++hits;
            ++cursor;
        }
        if (hits > 0) {
            counts[index_offset + static_cast<std::uint64_t>(i)] += hits;
        }
    }
}

}  // namespace aegisq
