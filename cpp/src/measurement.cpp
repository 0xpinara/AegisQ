#include "aegisq/measurement.hpp"

#include <random>
#include <stdexcept>

namespace aegisq {

std::vector<double> sorted_uniform_draws(std::uint64_t shots, std::uint64_t seed, double total) {
    if (total <= 0.0) {
        throw std::runtime_error("cannot sample from a state with zero probability mass");
    }
    std::vector<double> draws(static_cast<std::size_t>(shots));
    std::mt19937_64 engine(seed);
    std::uniform_real_distribution<double> uniform(0.0, total);
    for (auto& draw : draws) {
        draw = uniform(engine);
    }
    std::sort(draws.begin(), draws.end());
    return draws;
}

}  // namespace aegisq
