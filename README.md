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
- [x] Phase 11 — mapper evaluation on measured hardware
- [x] Phase 12 — OpenQASM subset front end
- [x] Phase 13 — memory estimator
- [x] Phase 14 — post-quantum identities (ML-KEM-768 / ML-DSA-65)
- [x] Phase 15 — secure job envelopes
- [x] Phase 16 — verified job execution and replay protection
- [x] Phase 17 — signed result provenance and Merkle verification
- [x] Phase 18 — security model documentation

Benchmark numbers in this README are generated from raw measurements under
`benchmarks/raw/` by [`scripts/generate_report.py`](scripts/generate_report.py);
none are typed by hand.

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

## Measured results

<!-- BENCHMARK-RESULTS:START -->

All figures below were measured on **Apple M2 (8 logical cores)**, macOS-15.6.1-arm64-arm-64bit, Open MPI v5.0.10, AegisQ 0.1.0 at commit `74a1a238513b`. They describe that host and are not a claim about cluster hardware.

### Communication-aware placement, 8 ranks, 20 qubits

| circuit | measured MPI bytes, default | measured MPI bytes, optimized | reduction | wall time change |
|---|---:|---:|---:|---:|
| qft | 981,467,136 | 125,829,120 | **87.2%** | -24.3% |
| grover | 1,879,048,192 | 520,093,696 | **72.3%** | -20.7% |
| random | 411,041,792 | 251,658,240 | **38.8%** | -20.0% |
| ising | 452,984,832 | 385,875,968 | **14.8%** | +1.1% |
| ghz | 25,165,824 | 25,165,824 | **0.0%** | -7.9% |

Not every circuit benefits: ghz shows no reduction, because its expensive qubits already sit well under the default placement. That is a result, not a gap — a heuristic that claimed a win on every circuit would be the suspicious one.

![Communication-aware placement](benchmarks/plots/mapping_comparison.png)

### Strong scaling (one thread per rank)

| circuit | qubits | ranks | wall time (s) | speedup | efficiency |
|---|---:|---:|---:|---:|---:|
| ising | 22 | 1 | 1.896 | 1.00x | 100% |
| ising | 22 | 2 | 1.031 | 1.84x | 92% |
| ising | 22 | 4 | 0.742 | 2.56x | 64% |
| ising | 22 | 8 | 0.793 | 2.39x | 30% |
| qft | 22 | 1 | 5.454 | 1.00x | 100% |
| qft | 22 | 2 | 2.959 | 1.84x | 92% |
| qft | 22 | 4 | 2.237 | 2.44x | 61% |
| qft | 22 | 8 | 2.319 | 2.35x | 29% |

![Strong scaling](benchmarks/plots/strong_scaling.png)

### Cost of the post-quantum layer

| operation | median | size |
|---|---:|---:|
| ML-DSA-65 keygen | 53.8 us | 1952 B |
| ML-DSA-65 sign | 101.8 us | 3309 B |
| ML-DSA-65 verify | 49.6 us | 3309 B |
| ML-KEM-768 decapsulate | 20.0 us | 32 B |
| ML-KEM-768 encapsulate | 17.5 us | 1088 B |
| ML-KEM-768 keygen | 16.9 us | 1184 B |
| pack a job bundle (end to end) | 3.1 ms | — |
| verify, decrypt and open it | 5.6 ms | — |
| envelope overhead, independent of circuit size | — | 7083 B |

For scale: the heaviest job measured here (grover, 20 qubits, 8 ranks) runs for 582 ms and moves 1792 MiB over MPI. Securing it costs 8.7 ms end to end and 6.9 KiB on the wire — 1.49% of the runtime. Only 189 us of that is lattice arithmetic; the rest is canonical serialisation and base64, which is where an optimisation would actually pay off.

### Why post-quantum cryptography, in one table

| search space | Grover oracle queries | classical expected | measured success |
|---:|---:|---:|---:|
| 4 | 1 | 2.5 | 100.0% |
| 8 | 2 | 4.5 | 95.4% |
| 16 | 3 | 8.5 | 96.6% |
| 32 | 4 | 16.5 | 99.9% |
| 64 | 6 | 32.5 | 99.7% |
| 128 | 8 | 64.5 | 99.6% |
| 256 | 12 | 128.5 | 100.0% |

Every row is a simulated run, not a formula: searching 256 items took 12 oracle queries where classical search averages 128.5, and the marked state was measured 100.0% of the time. That quadratic factor is why post-quantum guidance doubles symmetric key sizes rather than abandoning them — while Shor's exponential advantage is why RSA and elliptic curves are replaced outright.

![Grover query scaling](benchmarks/plots/grover_scaling.png)

### Cost model versus reality

In all **42 of 42** distributed configurations measured here, the runtime sent exactly the number of bytes the analytical cost model predicted.

Raw measurements: [`benchmarks/raw/`](benchmarks/raw/) · derived tables: [`benchmarks/processed/`](benchmarks/processed/) · methodology: [`docs/benchmark-methodology.md`](docs/benchmark-methodology.md)

<!-- BENCHMARK-RESULTS:END -->

## Running a circuit

```bash
# single process
aegisq run examples/bell.qasm --shots 1000

# distributed over four ranks
mpirun -np 4 aegisq run examples/ghz8.qasm --shots 1000

# with a communication-aware placement
mpirun -np 8 aegisq run qft --qubits 24 --shots 1024 --optimize

# what would a placement cost?
aegisq optimize random --qubits 20 --ranks 8 --option seed=3

# will it fit?
aegisq estimate --qubits 30 --ranks 4 --precision fp64
```

### Secure job submission

```bash
# once per identity
aegisq keys init-client pinar --directory keys
aegisq keys init-cluster courant --directory keys

# client side: encrypt to the cluster, sign with your key
aegisq secure-pack qft --qubits 24 --identity keys/pinar \
    --cluster keys/courant.public.json --ranks 8 --shots 1024 --output qft24.aqjob

# cluster side: verify, decrypt, run, sign the result
mpirun -np 8 aegisq secure-run qft24.aqjob --cluster keys/courant \
    --trusted keys/trusted --output qft24.aqresult

# anyone: check the signed execution record
aegisq verify-result qft24.aqresult --cluster keys/courant.public.json
```

Circuit input is a **documented subset** of OpenQASM (single register, the
twelve supported gates, terminal measurement); anything outside it is rejected
with a line number rather than ignored. The grammar is in
[docs/architecture.md](docs/architecture.md).

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
