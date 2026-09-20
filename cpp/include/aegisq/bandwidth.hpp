#pragma once

/// A memory-bandwidth reference for the local kernels.
///
/// State-vector simulation is bandwidth-bound: a single-qubit gate reads and
/// writes every amplitude once and does a handful of flops per element. Wall
/// time alone cannot say whether a kernel is good; it has to be compared
/// against what the machine can move.
///
/// The reference is a STREAM-style triad over the same data type the kernels
/// use, compiled with the same flags and threaded the same way, so the
/// comparison is between two numbers measured under the same conditions
/// rather than against a vendor figure.

#include <complex>
#include <cstddef>

namespace aegisq {

struct BandwidthSample {
    double seconds{0.0};
    std::size_t bytes{0};

    double gigabytes_per_second() const {
        return seconds > 0.0 ? static_cast<double>(bytes) / seconds / 1e9 : 0.0;
    }
};

/// STREAM triad, `a[i] = b[i] + scalar * c[i]`, over complex<double>.
///
/// Counts three streams (two read, one written), which is the conventional
/// accounting and matches how a gate kernel's traffic is counted below.
BandwidthSample stream_triad(std::size_t elements, int repeats = 5);

/// STREAM copy, `a[i] = b[i]`: two streams across two arrays.
BandwidthSample stream_copy(std::size_t elements, int repeats = 5);

/// In-place scale, `a[i] *= s`: two streams over *one* array.
///
/// This is the reference that matches a gate kernel's traffic shape. A gate
/// reads and writes the same array, which is not the same thing as copying
/// between two arrays: one array means half the pages and half the TLB
/// pressure, and a kernel can legitimately exceed a two-array copy.
BandwidthSample stream_scale_in_place(std::size_t elements, int repeats = 5);

}  // namespace aegisq
