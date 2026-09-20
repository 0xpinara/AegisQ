#include "aegisq/bandwidth.hpp"

#include <algorithm>
#include <chrono>
#include <vector>

#include "aegisq/kernels.hpp"

namespace aegisq {
namespace {

using Complex = std::complex<double>;

double elapsed_since(const std::chrono::steady_clock::time_point& start) {
    const std::chrono::duration<double> delta = std::chrono::steady_clock::now() - start;
    return delta.count();
}

}  // namespace

BandwidthSample stream_triad(std::size_t elements, int repeats) {
    std::vector<Complex> a(elements, Complex{0.0, 0.0});
    std::vector<Complex> b(elements, Complex{1.0, 0.5});
    std::vector<Complex> c(elements, Complex{2.0, -0.5});
    const Complex scalar{3.0, 0.0};

    // One untimed pass so the pages are resident and the caches are in a
    // steady state; a first touch would measure the allocator, not the bus.
    for (std::size_t i = 0; i < elements; ++i) {
        a[i] = b[i] + scalar * c[i];
    }

    double best = 0.0;
    for (int repeat = 0; repeat < std::max(1, repeats); ++repeat) {
        const auto started = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static) if (elements >= kernels::kParallelThreshold)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(elements); ++i) {
            const std::size_t index = static_cast<std::size_t>(i);
            a[index] = b[index] + scalar * c[index];
        }
        const double seconds = elapsed_since(started);
        best = (best == 0.0) ? seconds : std::min(best, seconds);
    }
    // Touch the result so the loop cannot be optimised away entirely.
    if (a[elements / 2] == Complex{1e308, 1e308}) {
        return BandwidthSample{0.0, 0};
    }

    // Three streams: b and c read, a written.
    return BandwidthSample{best, elements * sizeof(Complex) * 3};
}

BandwidthSample stream_scale_in_place(std::size_t elements, int repeats) {
    std::vector<Complex> a(elements, Complex{1.0, 0.5});
    const Complex scalar{1.000001, 0.0};

    for (std::size_t i = 0; i < elements; ++i) {
        a[i] *= scalar;
    }

    double best = 0.0;
    for (int repeat = 0; repeat < std::max(1, repeats); ++repeat) {
        const auto started = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static) if (elements >= kernels::kParallelThreshold)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(elements); ++i) {
            a[static_cast<std::size_t>(i)] *= scalar;
        }
        const double seconds = elapsed_since(started);
        best = (best == 0.0) ? seconds : std::min(best, seconds);
    }
    if (a[elements / 2] == Complex{1e308, 1e308}) {
        return BandwidthSample{0.0, 0};
    }

    // One array, read and written: the same traffic shape as a gate kernel.
    return BandwidthSample{best, elements * sizeof(Complex) * 2};
}

BandwidthSample stream_copy(std::size_t elements, int repeats) {
    std::vector<Complex> a(elements, Complex{0.0, 0.0});
    std::vector<Complex> b(elements, Complex{1.0, 0.5});

    for (std::size_t i = 0; i < elements; ++i) {
        a[i] = b[i];
    }

    double best = 0.0;
    for (int repeat = 0; repeat < std::max(1, repeats); ++repeat) {
        const auto started = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static) if (elements >= kernels::kParallelThreshold)
        for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(elements); ++i) {
            const std::size_t index = static_cast<std::size_t>(i);
            a[index] = b[index];
        }
        const double seconds = elapsed_since(started);
        best = (best == 0.0) ? seconds : std::min(best, seconds);
    }
    if (a[elements / 2] == Complex{1e308, 1e308}) {
        return BandwidthSample{0.0, 0};
    }

    return BandwidthSample{best, elements * sizeof(Complex) * 2};
}

}  // namespace aegisq
