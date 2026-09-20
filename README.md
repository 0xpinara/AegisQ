# AegisQ-HPC

**Post-Quantum Secure, Communication-Aware Distributed Quantum Simulation for HPC Clusters**

![status](https://img.shields.io/badge/status-research%20prototype-orange)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![c++](https://img.shields.io/badge/C%2B%2B-20-blue)
![mpi](https://img.shields.io/badge/MPI-OpenMPI%20%7C%20MPICH-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A distributed quantum state-vector simulator that partitions the state across
MPI ranks, **chooses the qubit-to-rank placement deliberately instead of by
accident**, measures every byte it sends, and protects job submission and
result provenance with NIST-standardised post-quantum cryptography.

The simulator is written from scratch in C++20 with MPI and OpenMP. Qiskit is
used **only** as a correctness oracle in the test suite — never by the
runtime, and never as a performance baseline (no simulator-versus-simulator
comparison is made here; see [docs/limitations.md](docs/limitations.md)).

> **Headline result.** On 8 ranks at 20 qubits, communication-aware placement
> removed **87.2% of measured MPI traffic** for a quantum Fourier transform
> (wall time −24.3%) and 72.3% for Grover — and left a GHZ chain untouched,
> because nothing there can be improved. The analytical cost model predicted
> the byte count *exactly* in all 42 distributed configurations measured.
> [Full numbers below](#measured-results), from raw data in
> [`benchmarks/raw/`](benchmarks/raw/).

```
     circuit ──► communication cost model ──► qubit placement search
                                                      │
        .aqjob  (ML-KEM-768 + ML-DSA-65 + AES-256-GCM)│
           │                                          ▼
           └─► verify ─► decrypt ─► ┌──────────── distributed runtime ────────────┐
                                    │ rank 0    rank 1    rank 2    rank 3        │
                                    │ shard     shard     shard     shard         │
                                    │      ↔ instrumented MPI exchange ↔          │
                                    └──────────────────┬──────────────────────────┘
                                                       ▼
                             .aqresult  (Merkle root + ML-DSA signature + metrics)
```

## What is implemented

| Area | Capability |
|---|---|
| Simulation | distributed state vector over MPI, OpenMP local kernels, fp64/fp32, twelve gates, terminal measurement |
| Placement | gate-role-aware cost model in bytes, exhaustive or heuristic search, applied by the runtime |
| Fusion | consecutive single-qubit runs multiplied into one unitary, exact including global phase, never increasing communication |
| Windowed placement | the assignment may change mid-circuit when the saving exceeds the cost of moving, planned by dynamic programming and executed as a circuit rewrite |
| Instrumentation | every MPI transfer timed and counted, per opcode, rank-local and world-reduced |
| Security | ML-KEM-768 / ML-DSA-65 job envelopes, replay protection, signed Merkle-committed results |
| Algorithms | GHZ, QFT, Trotterised Ising, Grover, Shor (factors 15 and 21), random circuits |
| Measurement | reproducible sweeps writing raw CSV; every figure and table derived from it, including kernel bandwidth against a machine reference |
| Validation | every distributed kernel checked against a single-process reference at 1/2/4/8 ranks, plus 100 randomised Qiskit comparisons per run |

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

All figures below were measured on **Apple M2 (8 logical cores)**, macOS-15.6.1-arm64-arm-64bit, Open MPI v5.0.10, AegisQ 0.1.0 across 7 commits (`54ec4cfa1c2e`, `5984b9d32bf6`, `74a1a238513b`, `a77b4e655ca1`, `b67b6064b543`, `c96b197e4e0b`, `dabb38de9c99`). They describe that host and are not a claim about cluster hardware.

> Some rows were recorded from a working tree with uncommitted changes, so they cannot be attributed to a commit with confidence. Re-run `./scripts/benchmark_local.sh` from a clean tree to replace them.

### Communication-aware placement, 8 ranks, 20 qubits

| circuit | measured MPI bytes, default | measured MPI bytes, optimized | reduction | wall time change |
|---|---:|---:|---:|---:|
| qft | 981,467,136 | 125,829,120 | **87.2%** | -31.5% |
| grover | 1,879,048,192 | 520,093,696 | **72.3%** | -17.8% |
| random | 411,041,792 | 251,658,240 | **38.8%** | -14.5% |
| ising | 452,984,832 | 385,875,968 | **14.8%** | -7.3% |
| ghz | 25,165,824 | 25,165,824 | **0.0%** | +20.7% |

Not every circuit benefits: ghz shows no reduction, because its expensive qubits already sit well under the default placement. That is a result, not a gap — a heuristic that claimed a win on every circuit would be the suspicious one.

![Communication-aware placement](benchmarks/plots/mapping_comparison.png)

### Three ways to spend less on the network

Placement decides *which* gates communicate. Fusion decides *how many times*. Windowed placement changes the assignment part-way through the circuit, when the phase structure makes the switch worth paying for. Measured at 8 ranks:

| circuit | baseline MPI bytes | fusion | static placement | windowed placement | placement + fusion |
|---|---:|---:|---:|---:|---:|
| qft | 981,467,136 | 0.0% | 87.2% | **95.7%** | 87.2% |
| grover | 1,879,048,192 | 16.1% | 72.3% | **87.1%** | 86.6% |
| random | 411,041,792 | 8.2% | 38.8% | 42.9% | **53.1%** |
| ising | 452,984,832 | 0.0% | 14.8% | **16.7%** | 14.8% |
| ghz | 25,165,824 | 0.0% | 0.0% | 0.0% | 0.0% |

Windowed placement beats the best single assignment on 4 of 5 circuits — for `qft`, 95.7% against 87.2% — by paying a few shard exchanges to re-assign qubits between phases. Where a circuit has no phase structure the planner declines to switch and the two coincide.

Placement and fusion are not independent either: for `random` the pair removes 53.1% where separately they remove 8.2% and 38.8%. Fusing first changes which placement is best, so the search finds a better one.

![Optimisation levers](benchmarks/plots/lever_comparison.png)

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

### Is the fallback search good enough?

The placement search is exhaustive while the candidate count fits a budget and falls back beyond it. Two facts make the fallback safer than "not guaranteed optimal" suggests.

Every cost rule except `swap` depends on one qubit's membership: a single-qubit gate costs if *its* qubit is global, a `cx` costs if *its target* is. `swap` costs if *either* operand is, and an OR is not a sum. So for a circuit without `swap` gates the objective is linear in the global set, and sorting the per-qubit costs gives the optimum outright — no search.

For the rest, measured: across 99 sampled configurations, 93 of which genuinely needed the heuristic, it matched the exhaustive optimum **99 times out of 99** (worst gap 0.00%) while running up to 40x faster.

### Are the local kernels any good?

Wall time alone cannot say. State-vector simulation is bandwidth-bound, so each kernel is compared against what the same machine achieves on an in-place scale of the same array — same type, same flags, same threading. At 8 threads:

| kernel | GB/s achieved | reference | fraction |
|---|---:|---:|---:|
| rz | 68.8 | 69.8 | 98% |
| h | 68.4 | 69.8 | 98% |
| cz | 40.9 | 69.8 | 59% |
| swap | 37.2 | 69.8 | 53% |
| cx | 36.2 | 69.8 | 52% |

The kernels that sweep the whole state — a general single-qubit gate and a diagonal one — run at 98% of that reference, which is to say they are memory-bound and there is little left to win. The ones that touch only part of the state plateau near 55%: they read and write a strided fraction of the array and leave about half the bandwidth unused. That is a concrete optimisation target rather than a mystery.

The communication cost model assumes only the local/global distinction matters, never which *local* position a qubit occupies. Measured, that holds for most kernels (`cx` varies by 3% across target positions) and fails for `cz`, which varies by 57%. The model is therefore right about network traffic and incomplete about local cost — stated here rather than left for a reader to discover.

![Local kernel bandwidth](benchmarks/plots/kernel_bandwidth.png)

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

For scale: the heaviest job measured here (grover, 20 qubits, 8 ranks) runs for 569 ms and moves 1792 MiB over MPI. Securing it costs 8.7 ms end to end and 6.9 KiB on the wire — 1.52% of the runtime. Only 189 us of that is lattice arithmetic; the rest is canonical serialisation and base64, which is where an optimisation would actually pay off.

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

In all **87 of 87** distributed configurations measured here, the runtime sent exactly the number of bytes the analytical cost model predicted.

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

## Documentation

| Document | Contents |
|---|---|
| [architecture.md](docs/architecture.md) | layer responsibilities, native core layout, OpenQASM subset |
| [simulator-theory.md](docs/simulator-theory.md) | state-vector conventions, gate set, why measurement is terminal |
| [distributed-algorithm.md](docs/distributed-algorithm.md) | partitioning, which gates communicate and how much |
| [optimizer.md](docs/optimizer.md) | cost model, search strategy, what the model cannot see |
| [pqc-protocol.md](docs/pqc-protocol.md) | envelope format, canonical bytes, result bundles |
| [security-model.md](docs/security-model.md) | assets, adversaries, exclusions, what a signature proves |
| [benchmark-methodology.md](docs/benchmark-methodology.md) | what is counted, thread policies, statistics |
| [reproducibility.md](docs/reproducibility.md) | what is deterministic and what is not |
| [limitations.md](docs/limitations.md) | every constraint of this implementation, in one place |
| [paper/main.tex](paper/main.tex) | technical report; tables are generated from the measured data |

## Scope and honesty statement

AegisQ-HPC is a research prototype, not a production security product.

- Post-quantum primitives come from [liboqs](https://openquantumsafe.org/);
  no cryptography is invented here.
- A signed execution record authenticates origin and detects tampering. It is
  **not** a proof that a computation was performed correctly.
- The Grover and Shor demonstrations run at educational problem sizes and make
  no claim about cryptographically relevant key or modulus sizes.
- No quantum advantage is claimed anywhere in this repository.
- All measurements come from one laptop with shared-memory MPI; see
  [docs/limitations.md](docs/limitations.md) for what that does and does not
  support.

## Licence

MIT — see [LICENSE](LICENSE).
