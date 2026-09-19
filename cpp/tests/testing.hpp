#pragma once

// Minimal assertion helpers so the C++ test suite has no external dependency.

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace aegisq::testing {

inline int failures = 0;

inline void report(const char* file, int line, const std::string& message) {
    std::fprintf(stderr, "FAIL %s:%d  %s\n", file, line, message.c_str());
    ++failures;
}

inline int summary(const char* suite) {
    if (failures == 0) {
        std::printf("PASS %s\n", suite);
        return 0;
    }
    std::fprintf(stderr, "%d assertion(s) failed in %s\n", failures, suite);
    return 1;
}

inline bool close(double a, double b, double tol) {
    return std::fabs(a - b) <= tol;
}

inline bool close(std::complex<double> a, std::complex<double> b, double tol) {
    return std::abs(a - b) <= tol;
}

}  // namespace aegisq::testing

#define AEGISQ_CHECK(cond)                                                  \
    do {                                                                    \
        if (!(cond)) {                                                      \
            ::aegisq::testing::report(__FILE__, __LINE__, "check: " #cond); \
        }                                                                   \
    } while (0)

#define AEGISQ_CHECK_CLOSE(a, b, tol)                                                  \
    do {                                                                               \
        if (!::aegisq::testing::close((a), (b), (tol))) {                              \
            ::aegisq::testing::report(__FILE__, __LINE__, "not close: " #a " vs " #b); \
        }                                                                              \
    } while (0)
