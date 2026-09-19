# Changelog

All notable changes to AegisQ-HPC are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project foundation: CMake/C++20 core skeleton, pybind11 extension plumbing,
  Python package layout, pytest configuration, Makefile, Dockerfile and CI.
- `aegisq --version` and `aegisq doctor` environment diagnostics.
- Circuit IR: `Gate`/`Circuit` with validation, depth analysis, inversion,
  composition and canonical dictionary serialisation.
- Supported gate set: `x y z h s t rx ry rz cx cz swap` plus terminal
  measurement, each annotated with the diagonal/control structure the
  communication cost model needs.
- NumPy reference state-vector simulator with seeded, reproducible sampling.
- `Simulator` front end with a backend registry and `SimulationResult`.
- C++20 simulation core: templated state vector (fp64/fp32), paired-index
  local kernels with OpenMP threading, gate/circuit containers and a
  partition-independent shot sampler.
- pybind11 bindings exposing `StateVectorF64`, `StateVectorF32`, `Gate` and
  `Circuit`, plus the `cpp` simulator backend.
