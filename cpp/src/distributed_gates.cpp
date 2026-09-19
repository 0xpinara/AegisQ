/// Distributed state-vector construction and gate execution.
///
/// Everything here is driven by `DistributedLayout`: whether a gate needs
/// communication at all is a property of where its operands sit, not of the
/// gate alone.

#include <algorithm>
#include <stdexcept>

#include "aegisq/kernels.hpp"
#include "aegisq/statevector.hpp"

namespace aegisq {

template <typename Real>
DistributedStateVectorT<Real>::DistributedStateVectorT(int num_qubits)
    : DistributedStateVectorT(num_qubits, std::vector<int>{}) {}

template <typename Real>
DistributedStateVectorT<Real>::DistributedStateVectorT(int num_qubits,
                                                       std::vector<int> logical_to_position)
    : layout_(num_qubits, MpiContext::instance().world_size(), MpiContext::instance().rank(),
              std::move(logical_to_position)),
      local_(layout_.local_state_size(), Amplitude{0, 0}) {
    reset();
}

template <typename Real>
void DistributedStateVectorT<Real>::reset() {
    std::fill(local_.begin(), local_.end(), Amplitude{0, 0});
    // |0...0> is physical index 0 regardless of the qubit mapping, and index 0
    // lives on rank 0 by construction.
    if (layout_.rank() == 0) {
        local_[0] = Amplitude{1, 0};
    }
    metrics_ = LocalMetrics{};
}

template <typename Real>
double DistributedStateVectorT<Real>::local_squared_norm() const {
    return kernels::squared_norm(local_.data(), local_.size());
}

template <typename Real>
double DistributedStateVectorT<Real>::norm() const {
    double local_total = local_squared_norm();
#if AEGISQ_HAVE_MPI
    if (layout_.world_size() > 1) {
        double global_total = 0.0;
        MPI_Allreduce(&local_total, &global_total, 1, MPI_DOUBLE, MPI_SUM,
                      MpiContext::instance().comm());
        return global_total;
    }
#endif
    return local_total;
}

template <typename Real>
std::vector<std::complex<double>> DistributedStateVectorT<Real>::gather() const {
    const std::size_t local_size = local_.size();
    std::vector<std::complex<double>> physical(local_size *
                                               static_cast<std::size_t>(layout_.world_size()));

    std::vector<std::complex<double>> local_as_double(local_size);
    for (std::size_t i = 0; i < local_size; ++i) {
        local_as_double[i] = std::complex<double>(static_cast<double>(local_[i].real()),
                                                  static_cast<double>(local_[i].imag()));
    }

#if AEGISQ_HAVE_MPI
    if (layout_.world_size() > 1) {
        MPI_Allgather(local_as_double.data(), static_cast<int>(local_size), MPI_C_DOUBLE_COMPLEX,
                      physical.data(), static_cast<int>(local_size), MPI_C_DOUBLE_COMPLEX,
                      MpiContext::instance().comm());
    } else {
        physical = local_as_double;
    }
#else
    physical = local_as_double;
#endif

    if (layout_.is_identity_mapping()) {
        return physical;
    }

    // Re-order into logical basis order so callers can compare against the
    // single-process reference without knowing the placement.
    std::vector<std::complex<double>> logical(physical.size());
    for (std::uint64_t index = 0; index < physical.size(); ++index) {
        logical[layout_.to_logical_index(index)] = physical[index];
    }
    return logical;
}

template class DistributedStateVectorT<double>;
template class DistributedStateVectorT<float>;

}  // namespace aegisq
