#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include "aegisq/circuit.hpp"
#include "aegisq/distributed_layout.hpp"
#include "aegisq/gate.hpp"
#include "aegisq/measurement.hpp"
#include "aegisq/mpi_context.hpp"
#include "aegisq/statevector.hpp"
#include "aegisq/version.hpp"

namespace py = pybind11;

namespace {

/// Expose one precision of the state vector under a distinct Python name.
template <typename Real>
void bind_statevector(py::module_& m, const char* name) {
    using SV = aegisq::StateVectorT<Real>;
    py::class_<SV>(m, name)
        .def(py::init<int>(), py::arg("num_qubits"))
        .def_property_readonly("num_qubits", &SV::num_qubits)
        .def_property_readonly("size", &SV::size)
        .def("reset", &SV::reset)
        .def("norm", &SV::norm)
        .def("apply_x", &SV::apply_x, py::arg("qubit"))
        .def("apply_y", &SV::apply_y, py::arg("qubit"))
        .def("apply_z", &SV::apply_z, py::arg("qubit"))
        .def("apply_h", &SV::apply_h, py::arg("qubit"))
        .def("apply_s", &SV::apply_s, py::arg("qubit"))
        .def("apply_t", &SV::apply_t, py::arg("qubit"))
        .def("apply_rx", &SV::apply_rx, py::arg("qubit"), py::arg("theta"))
        .def("apply_ry", &SV::apply_ry, py::arg("qubit"), py::arg("theta"))
        .def("apply_rz", &SV::apply_rz, py::arg("qubit"), py::arg("theta"))
        .def("apply_cnot", &SV::apply_cnot, py::arg("control"), py::arg("target"))
        .def("apply_cz", &SV::apply_cz, py::arg("a"), py::arg("b"))
        .def("apply_swap", &SV::apply_swap, py::arg("a"), py::arg("b"))
        .def("apply_gate", &SV::apply_gate, py::arg("gate"))
        .def("apply_circuit", &SV::apply_circuit, py::arg("circuit"))
        .def(
            "measure_all",
            [](const SV& self, std::uint64_t shots, std::uint64_t seed) {
                const aegisq::MeasurementResult result = self.measure_all(shots, seed);
                py::dict counts;
                for (const auto& [index, count] : result.counts) {
                    counts[py::int_(index)] = py::int_(count);
                }
                return counts;
            },
            py::arg("shots"), py::arg("seed"))
        .def("probabilities",
             [](const SV& self) {
                 const std::vector<double> probs = self.probabilities();
                 return py::array_t<double>(static_cast<py::ssize_t>(probs.size()), probs.data());
             })
        .def(
            "amplitudes",
            [](const SV& self) {
                // Copy into a NumPy array; the caller owns the result.
                return py::array_t<std::complex<Real>>(static_cast<py::ssize_t>(self.size()),
                                                       self.data());
            },
            "Final state as a NumPy array of complex amplitudes.")
        .def("set_amplitudes",
             [](SV& self,
                py::array_t<std::complex<Real>, py::array::c_style | py::array::forcecast> values) {
                 const auto view = values.template unchecked<1>();
                 std::vector<std::complex<Real>> buffer(static_cast<std::size_t>(view.shape(0)));
                 for (py::ssize_t i = 0; i < view.shape(0); ++i) {
                     buffer[static_cast<std::size_t>(i)] = view(i);
                 }
                 self.set_amplitudes(buffer);
             })
        .def_property_readonly("gates_applied",
                               [](const SV& self) { return self.metrics().gates_applied; })
        .def_property_readonly("compute_seconds",
                               [](const SV& self) { return self.metrics().compute_seconds; })
        .def("reset_metrics", &SV::reset_metrics);
}

/// Expose one precision of the distributed state vector.
template <typename Real>
void bind_distributed_statevector(py::module_& m, const char* name) {
    using DSV = aegisq::DistributedStateVectorT<Real>;
    py::class_<DSV>(m, name)
        .def(py::init<int>(), py::arg("num_qubits"))
        .def(py::init<int, std::vector<int>>(), py::arg("num_qubits"),
             py::arg("logical_to_position"))
        .def_property_readonly("num_qubits", &DSV::num_qubits)
        .def_property_readonly("rank", &DSV::rank)
        .def_property_readonly("world_size", &DSV::world_size)
        .def_property_readonly("local_size", &DSV::local_size)
        .def_property_readonly("layout", &DSV::layout, py::return_value_policy::reference_internal)
        .def("reset", &DSV::reset)
        .def("apply_x", &DSV::apply_x, py::arg("qubit"))
        .def("apply_y", &DSV::apply_y, py::arg("qubit"))
        .def("apply_z", &DSV::apply_z, py::arg("qubit"))
        .def("apply_h", &DSV::apply_h, py::arg("qubit"))
        .def("apply_s", &DSV::apply_s, py::arg("qubit"))
        .def("apply_t", &DSV::apply_t, py::arg("qubit"))
        .def("apply_rx", &DSV::apply_rx, py::arg("qubit"), py::arg("theta"))
        .def("apply_ry", &DSV::apply_ry, py::arg("qubit"), py::arg("theta"))
        .def("apply_rz", &DSV::apply_rz, py::arg("qubit"), py::arg("theta"))
        .def("apply_cnot", &DSV::apply_cnot, py::arg("control"), py::arg("target"))
        .def("apply_cz", &DSV::apply_cz, py::arg("a"), py::arg("b"))
        .def("apply_swap", &DSV::apply_swap, py::arg("a"), py::arg("b"))
        .def("apply_gate", &DSV::apply_gate, py::arg("gate"))
        .def("apply_circuit", &DSV::apply_circuit, py::arg("circuit"))
        .def("norm", &DSV::norm)
        .def("local_squared_norm", &DSV::local_squared_norm)
        .def("gather",
             [](const DSV& self) {
                 const std::vector<std::complex<double>> full = self.gather();
                 return py::array_t<std::complex<double>>(static_cast<py::ssize_t>(full.size()),
                                                          full.data());
             })
        .def("local_amplitudes",
             [](const DSV& self) {
                 return py::array_t<std::complex<Real>>(static_cast<py::ssize_t>(self.local_size()),
                                                        self.local_amplitudes().data());
             })
        .def_property_readonly("gates_applied",
                               [](const DSV& self) { return self.metrics().gates_applied; })
        .def_property_readonly("compute_seconds",
                               [](const DSV& self) { return self.metrics().compute_seconds; })
        .def("reset_metrics", &DSV::reset_metrics);
}

}  // namespace

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

    py::enum_<aegisq::OpCode>(m, "OpCode")
        .value("X", aegisq::OpCode::X)
        .value("Y", aegisq::OpCode::Y)
        .value("Z", aegisq::OpCode::Z)
        .value("H", aegisq::OpCode::H)
        .value("S", aegisq::OpCode::S)
        .value("T", aegisq::OpCode::T)
        .value("RX", aegisq::OpCode::RX)
        .value("RY", aegisq::OpCode::RY)
        .value("RZ", aegisq::OpCode::RZ)
        .value("CX", aegisq::OpCode::CX)
        .value("CZ", aegisq::OpCode::CZ)
        .value("SWAP", aegisq::OpCode::SWAP);

    m.def("opcode_from_name", &aegisq::opcode_from_name, py::arg("name"));
    m.def("gate_is_diagonal", &aegisq::gate_is_diagonal, py::arg("opcode"));
    m.def("gate_arity", &aegisq::gate_arity, py::arg("opcode"));

    py::class_<aegisq::Gate>(m, "Gate")
        .def(py::init([](const std::string& opcode, const std::vector<int>& qubits, double param) {
                 const aegisq::OpCode code = aegisq::opcode_from_name(opcode);
                 if (qubits.size() != static_cast<std::size_t>(aegisq::gate_arity(code))) {
                     throw std::invalid_argument("wrong number of qubits for " + opcode);
                 }
                 return qubits.size() == 1 ? aegisq::Gate::one(code, qubits[0], param)
                                           : aegisq::Gate::two(code, qubits[0], qubits[1]);
             }),
             py::arg("opcode"), py::arg("qubits"), py::arg("param") = 0.0)
        .def_readonly("opcode", &aegisq::Gate::opcode)
        .def_readonly("param", &aegisq::Gate::param)
        .def_property_readonly("qubits",
                               [](const aegisq::Gate& g) {
                                   std::vector<int> qs;
                                   for (int i = 0; i < aegisq::gate_arity(g.opcode); ++i) {
                                       qs.push_back(g.qubits[static_cast<std::size_t>(i)]);
                                   }
                                   return qs;
                               })
        .def("__repr__",
             [](const aegisq::Gate& g) { return "<Gate " + aegisq::to_string(g) + ">"; });

    py::class_<aegisq::Circuit>(m, "Circuit")
        .def(py::init<int>(), py::arg("num_qubits"))
        .def("add", py::overload_cast<const aegisq::Gate&>(&aegisq::Circuit::add), py::arg("gate"),
             py::return_value_policy::reference_internal)
        .def_property_readonly("num_qubits", &aegisq::Circuit::num_qubits)
        .def_property_readonly("size", &aegisq::Circuit::size)
        .def("depth", &aegisq::Circuit::depth)
        .def("__len__", &aegisq::Circuit::size)
        .def("__repr__", &aegisq::Circuit::to_string);

    // ---- MPI -------------------------------------------------------------
    m.def("mpi_compiled", &aegisq::MpiContext::compiled_with_mpi,
          "True when the core was built against MPI.");
    m.def(
        "mpi_rank", []() { return aegisq::MpiContext::instance().rank(); },
        "Rank of this process (initialises MPI on first call).");
    m.def(
        "mpi_world_size", []() { return aegisq::MpiContext::instance().world_size(); },
        "Number of ranks in the world communicator.");
    m.def(
        "mpi_barrier", []() { aegisq::MpiContext::instance().barrier(); },
        "Synchronise all ranks.");
    m.def("mpi_library_version", &aegisq::MpiContext::library_version);
    m.def("mpi_finalize", &aegisq::MpiContext::finalize,
          "Shut MPI down; idempotent and also registered with atexit.");

    m.def("is_power_of_two", &aegisq::is_power_of_two, py::arg("value"));
    m.def("log2_exact", &aegisq::log2_exact, py::arg("value"));

    py::class_<aegisq::DistributedLayout>(m, "DistributedLayout")
        .def(py::init<int, int, int>(), py::arg("num_qubits"), py::arg("world_size"),
             py::arg("rank"))
        .def(py::init<int, int, int, std::vector<int>>(), py::arg("num_qubits"),
             py::arg("world_size"), py::arg("rank"), py::arg("logical_to_position"))
        .def_property_readonly("num_qubits", &aegisq::DistributedLayout::num_qubits)
        .def_property_readonly("world_size", &aegisq::DistributedLayout::world_size)
        .def_property_readonly("rank", &aegisq::DistributedLayout::rank)
        .def_property_readonly("num_global_qubits", &aegisq::DistributedLayout::num_global_qubits)
        .def_property_readonly("num_local_qubits", &aegisq::DistributedLayout::num_local_qubits)
        .def_property_readonly("local_state_size", &aegisq::DistributedLayout::local_state_size)
        .def("local_state_bytes", &aegisq::DistributedLayout::local_state_bytes,
             py::arg("amplitude_bytes"))
        .def("position", &aegisq::DistributedLayout::position, py::arg("logical_qubit"))
        .def("is_local", &aegisq::DistributedLayout::is_local, py::arg("logical_qubit"))
        .def("is_global", &aegisq::DistributedLayout::is_global, py::arg("logical_qubit"))
        .def("global_position", &aegisq::DistributedLayout::global_position,
             py::arg("logical_qubit"))
        .def("global_bit_for_rank", &aegisq::DistributedLayout::global_bit_for_rank,
             py::arg("logical_qubit"), py::arg("rank"))
        .def("global_bit", &aegisq::DistributedLayout::global_bit, py::arg("logical_qubit"))
        .def("partner_rank_for_global_qubit",
             &aegisq::DistributedLayout::partner_rank_for_global_qubit, py::arg("logical_qubit"))
        .def("partner_rank_of", &aegisq::DistributedLayout::partner_rank_of,
             py::arg("logical_qubit"), py::arg("rank"))
        .def("local_qubits", &aegisq::DistributedLayout::local_qubits)
        .def("global_qubits", &aegisq::DistributedLayout::global_qubits)
        .def_property_readonly("logical_to_position",
                               &aegisq::DistributedLayout::logical_to_position)
        .def("is_identity_mapping", &aegisq::DistributedLayout::is_identity_mapping)
        .def("physical_index", &aegisq::DistributedLayout::physical_index, py::arg("local_index"))
        .def("physical_index_for_rank", &aegisq::DistributedLayout::physical_index_for_rank,
             py::arg("rank"), py::arg("local_index"))
        .def("to_logical_index", &aegisq::DistributedLayout::to_logical_index,
             py::arg("physical_index"))
        .def("to_physical_index", &aegisq::DistributedLayout::to_physical_index,
             py::arg("logical_index"))
        .def("__repr__", &aegisq::DistributedLayout::to_string);

    bind_statevector<double>(m, "StateVectorF64");
    bind_statevector<float>(m, "StateVectorF32");
    bind_distributed_statevector<double>(m, "DistributedStateVectorF64");
    bind_distributed_statevector<float>(m, "DistributedStateVectorF32");
}
