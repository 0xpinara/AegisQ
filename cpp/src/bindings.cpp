#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "aegisq/version.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_aegisq_core, m) {
    m.doc() = "AegisQ-HPC native state-vector simulation core";

    m.attr("__version__") = aegisq::version();

    m.def("version", &aegisq::version, "Core library version string.");
    m.def("compiler", &aegisq::compiler, "Compiler used to build the core library.");
    m.def("has_mpi", &aegisq::has_mpi, "Whether the core was built against MPI.");
    m.def("has_openmp", &aegisq::has_openmp, "Whether local kernels were built with OpenMP.");
    m.def("max_threads", &aegisq::max_threads, "OpenMP thread count available to local kernels.");

    m.def(
        "build_info",
        []() {
            py::dict info;
            info["version"] = aegisq::version();
            info["compiler"] = aegisq::compiler();
            info["mpi"] = aegisq::has_mpi();
            info["openmp"] = aegisq::has_openmp();
            info["max_threads"] = aegisq::max_threads();
            return info;
        },
        "Dictionary describing how the native core was compiled.");
}
