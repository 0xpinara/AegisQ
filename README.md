# AegisQ-HPC

**Post-Quantum Secure, Communication-Aware Distributed Quantum Simulation for HPC Clusters**

![status](https://img.shields.io/badge/status-research%20prototype-orange)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![c++](https://img.shields.io/badge/C%2B%2B-20-blue)
![mpi](https://img.shields.io/badge/MPI-OpenMPI%20%7C%20MPICH-blue)
![license](https://img.shields.io/badge/license-MIT-green)

AegisQ-HPC is a research prototype that studies the intersection of quantum
circuit simulation, distributed-memory high-performance computing, and
post-quantum secure job provenance.

The simulator is implemented from scratch: state vectors are partitioned
across MPI ranks by AegisQ's own runtime. Qiskit is used **only** as a
correctness oracle in the test suite, and external simulators (Qiskit Aer,
CUDA-Q) are used **only** as optional benchmark references.

## Research questions

| | Question |
|---|---|
| **RQ1** | Can communication-aware qubit placement reduce MPI communication volume and wall-clock time for distributed state-vector simulation? |
| **RQ2** | What overhead does authenticating and encrypting HPC quantum jobs and results with standardised post-quantum cryptography introduce? |
| **RQ3** | How do small Grover and Shor experiments illustrate the cryptographic motivation for post-quantum cryptography? |

## Why the state vector must be distributed

An `n`-qubit state vector holds `2^n` complex amplitudes. In double precision
each amplitude costs 16 bytes:

| Qubits | State size (fp64) |
|---:|---:|
| 26 | 1 GiB |
| 30 | 16 GiB |
| 32 | 64 GiB |
| 40 | 16 TiB |

Beyond a single machine's memory the state must be partitioned, and
partitioning turns some quantum gates into network operations. Which gates
those are depends on the qubit-to-rank mapping — and that is the optimisation
problem at the centre of this project.

## Project status

This repository is built in phases; only what is checked below is implemented.

- [x] Phase 0 — project foundation, build system, CI, `aegisq doctor`
- [x] Phase 1 — NumPy reference simulator
- [x] Phase 2 — C++20 single-process state-vector engine
- [x] Phase 3 — randomised Qiskit cross-validation
- [x] Phase 4 — distributed MPI state layout
- [x] Phase 5 — local and diagonal distributed gates
- [x] Phase 6 — global non-diagonal single-qubit gates
- [x] Phase 7 — topology-aware distributed CNOT
- [x] Phase 8 — communication profiler
- [x] Phase 9 — gate-aware communication cost model
- [x] Phase 10 — static communication-aware mapper
- [ ] Phase 11 — mapper evaluation on measured hardware
- [ ] Phase 12 — OpenQASM subset front end
- [ ] Phase 13 — memory estimator
- [ ] Phase 14 — post-quantum identities (ML-KEM-768 / ML-DSA-65)
- [ ] Phase 15 — secure job envelopes
- [ ] Phase 16 — verified job execution and replay protection
- [ ] Phase 17 — signed result provenance and Merkle verification
- [ ] Phase 18 — security model documentation

No benchmark numbers appear in this README until they have been measured on a
described host and committed as raw data under `benchmarks/raw/`.

## Quick start

```bash
git clone https://github.com/0xpinara/AegisQ-HPC.git
cd AegisQ-HPC

python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,crypto,benchmark,validation]"

make build      # configure + compile the C++20 core and Python extension
make test       # C++ unit tests and single-process Python tests
aegisq doctor   # report MPI / OpenMP / liboqs / Qiskit availability
```

### Requirements

| Component | Purpose | Required |
|---|---|---|
| C++20 compiler, CMake ≥ 3.24 | native simulation core | yes |
| Python 3.11+, NumPy | front end, reference backend | yes |
| MPI (OpenMPI or MPICH) | distributed runtime | for multi-rank execution |
| OpenMP | threaded local kernels | optional |
| liboqs-python | ML-KEM-768 / ML-DSA-65 | for the secure job layer |
| Qiskit | correctness oracle in tests | optional |
| CUDA | experimental GPU work | optional, not required |

## Repository layout

```
cpp/          C++20 simulation core, MPI runtime, pybind11 bindings
aegisq/       Python package: circuits, runtime, compiler, secure, provenance
tests/        unit, integration, MPI, crypto and cross-validation suites
benchmarks/   raw measurements, processed tables, generated plots
docs/         architecture, algorithms, protocol and methodology notes
paper/        technical report sources
slurm/        cluster job scripts
```

## Scope and honesty statement

AegisQ-HPC is a research prototype, not a production security product.

- Post-quantum primitives come from [liboqs](https://openquantumsafe.org/);
  no cryptography is invented here.
- A signed execution record authenticates origin and detects tampering. It is
  **not** a proof that a computation was performed correctly.
- The Grover and Shor demonstrations run at educational problem sizes and make
  no claim about cryptographically relevant key or modulus sizes.
- No quantum advantage is claimed anywhere in this repository.

## Licence

MIT — see [LICENSE](LICENSE).
