# Changelog

All notable changes to AegisQ-HPC are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Separable-objective fast path** in the placement search. Every cost rule
  except `swap` depends on a single qubit's membership of the global set, so a
  circuit without `swap` gates has a linear objective and the optimum is a
  sort rather than a search. Reported as `linear (separable objective)` and
  marked optimal, because it is.
- **`aegisq benchmark placement`**: measures the fallback search against the
  exhaustive optimum. Across 99 sampled configurations at 4, 8 and 16 ranks,
  93 of which needed the heuristic, it matched the optimum every time while
  running 3-40x faster.
- **Local kernel bandwidth measurement** (`aegisq benchmark kernels`): each
  kernel is timed and expressed as achieved GB/s against a STREAM-style
  reference measured the same way, including an in-place shape that matches a
  gate's traffic. Full-sweep kernels reach 98% of it; partial-sweep kernels
  plateau near 52–59%. The sweep over target positions also tests the cost
  model's assumption that local position does not matter — true for most
  kernels, false for `cz` at a 57% spread.
- **Windowed placement** (`aegisq.compiler.dynamic_mapper`): the qubit
  assignment may change part-way through a circuit when the phase structure
  pays for the switch. Planned by a dynamic program over windows and executed
  as a circuit rewrite, so the runtime is unchanged and every byte the plan
  spends is measured by the existing profiler. Beats the best static
  assignment on four of five benchmark families; 95.7% of baseline traffic
  removed for the QFT against 87.2% static.
- **Gate fusion** (`aegisq.compiler.fusion`): consecutive single-qubit gates
  are multiplied into one unitary, so a run on a global qubit costs one shard
  exchange instead of one per gate. Exact including global phase, never
  increases communication, and a fused run of diagonal gates stays diagonal
  and therefore free.
- A `u` opcode carrying an arbitrary single-qubit unitary as its four complex
  entries. Diagonality is determined per instruction rather than per opcode,
  in both the Python and C++ layers. It is an internal representation with no
  OpenQASM form, and the emitter says so rather than writing something lossy.
- Measured 2x2 experiment over placement and fusion (`--levers
  placement,fusion`), with a table and plot. The levers are not additive: for
  random circuits, fusion alone removes 8.2%, placement alone 38.8%, and the
  combination 53.1%, because fusing changes which placement is best.
- Property-based tests (Hypothesis) covering backend agreement, norm
  preservation, circuit and QASM round trips, the optimiser's fast scorer,
  canonical serialisation and Merkle audit paths. Budgets are profile-driven:
  a fast default locally, `--hypothesis-profile=deep` (2000 examples per
  property) in CI.
- Exhaustive distributed correctness tests at the smallest legal shard
  (one or two local qubits per rank), where index arithmetic degenerates.
- Tests for the chunked-transfer path, reachable now that the per-call element
  limit is adjustable at runtime.

### Changed
- `send_calls`/`receive_calls` now count **physical** MPI calls while
  `pairwise_exchanges` counts **logical** shard exchanges. They were
  previously identical, which made one of them meaningless and would have
  under-reported message counts for shards above the int-typed MPI limit.
- Shot and seed arguments are validated once in `Simulator.run`, so every
  backend rejects them with the same message. The native backend previously
  surfaced a pybind11 type error for a negative shot count.

### Fixed
- OpenQASM gate names are now case-sensitive, matching the treatment of
  declaration keywords; the error names the lowercase spelling.
- A malformed qubit index (`q[-1]`, `q[x]`) is reported as malformed instead
  of being blamed on register-wide application.

## [0.1.0] — 2026-09-19

First complete implementation: distributed simulation, communication-aware
placement, post-quantum job security and the measurements behind all three
research questions.

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
- Randomised cross-validation against Qiskit as an external oracle: 100 seeded
  circuits per run plus a slower 9-qubit, depth-30 sweep, covering both the
  reference and native backends.
- Distributed MPI layout: `DistributedLayout` with an explicit
  logical-qubit-to-physical-position permutation, partner-rank arithmetic and
  physical/logical index translation.
- `DistributedStateVectorT<Real>` with rank-local shards, allreduced norm and
  a test-only `gather`, plus lazy MPI initialisation so `import aegisq` never
  starts a communicator on its own.
- MPI test suite launched by `scripts/run_mpi_tests.sh` at 1, 2 and 4 ranks.
- Distributed execution of every gate placement that needs no communication:
  local gates, diagonal gates on global qubits (one scalar multiply for the
  whole shard), `cz` in all four placements, and `cx` with a global control
  and a local target. Placements that still require an exchange raise a
  precise error instead of producing a wrong state.
- Global non-diagonal single-qubit gates (`x`, `y`, `h`, `rx`, `ry`) through a
  symmetric MPI_Sendrecv shard exchange with `rank ^ (1 << global_position)`,
  a reused exchange buffer and chunking so element counts stay inside the
  int-typed MPI interface.
- Distributed CX in all four placement cases and SWAP with global operands.
  Placements where only half the shard participates pack the selected
  amplitudes into a contiguous buffer with a strided block copy, halving the
  bytes on the wire relative to a whole-shard exchange.
- `CommunicationProfiler`: every project-controlled MPI transfer is timed and
  counted (sends, receives, pairwise exchanges, bytes each way, collectives),
  with a per-opcode breakdown and a world-reduced view that sums byte counts
  and maximises times. Byte counts are pinned to closed-form expectations in
  the MPI suite.
- `CommunicationCostModel`: gate-role-aware prediction of bytes and messages
  for a candidate qubit placement, expressed in bytes so it can be checked
  against measured counters. Verified equal to measurement on the distributed
  runtime for random circuits, custom placements and fp32 shards.
- Per-opcode metrics are now reduced across ranks as well, guarded by a key-set
  agreement check so a mismatch degrades to rank-local numbers instead of
  deadlocking.
- Benchmark circuit families: GHZ, QFT (controlled phases decomposed to
  rz/cx), Trotterised transverse-field Ising, Grover with a Toffoli-ladder
  multi-controlled phase flip, and seeded random circuits.
- `StaticCommunicationMapper`: exhaustive search over C(n, p) placements when
  affordable, greedy plus pairwise local search beyond a candidate budget,
  with the result labelled accordingly. The chosen mapping is handed to the
  runtime, not merely reported.
- `aegisq optimize` CLI with a human-readable before/after comparison and a
  `--json` mode.
- OpenQASM subset front end (`parse_qasm`, `to_qasm`) that rejects anything
  outside the supported grammar with a line number, plus `examples/*.qasm`.
- Distributed measurement: rank-offset probability scan with MPI_Exscan and an
  allgathered count map, producing counts identical to a single-process run
  for the same seed.
- `aegisq run`, which partitions automatically under mpirun and can apply a
  communication-aware placement with `--optimize`.
- `aegisq estimate`: state size, per-rank shard and the *peak* working set
  including the exchange and packing buffers the runtime allocates, compared
  against this host's RAM.
- `aegisq doctor` additionally reports the MPI library string from the core.
- Benchmark harness: `aegisq benchmark strong|weak|mapping|report`, raw
  append-only CSV with full provenance per row, warm-up discarding,
  best-of-repeats statistics, explicit thread policies, plots generated only
  from raw data, and `scripts/generate_report.py` to keep the README's numbers
  derived rather than typed.
- Slurm scripts for strong scaling, weak scaling, the placement experiment and
  secure job execution.
- First measured results committed under `benchmarks/`.
- Post-quantum identities: ML-KEM-768 and ML-DSA-65 key pairs from liboqs,
  written with owner-only permissions, fingerprints recomputed on load so a
  doctored identity file cannot advertise a fingerprint it does not have.
- Canonical JSON serialisation for everything signed or hashed, including a
  parser that rejects non-canonical bytes.
- HKDF-SHA256 key derivation bound to protocol version and job id, and
  AES-256-GCM authenticated encryption.
- `aegisq keys init-client|init-cluster|fingerprint`.
- Signed, encrypted `.aqjob` bundles: the public header is the AEAD associated
  data and the ML-DSA signature covers header and ciphertext together, so any
  edit breaks decryption as well as the signature. The manifest and circuit
  are encrypted together so parameters cannot be separated from the program.
- `aegisq secure-pack` and `aegisq secure-inspect` (metadata without
  decrypting, and without needing any key).
- Replay protection: a locked, atomically-replaced state file of accepted job
  identifiers, with honest documentation of its limits (per-installation
  scope, flock semantics on NFS, opt-in pruning).
- `aegisq secure-run`: structure, trusted-key lookup, signature, addressee,
  replay state, decryption and circuit hash — in that order — then execution.
  Under mpirun every rank refuses a replayed job while only rank 0 records it,
  so the ranks cannot disagree and deadlock.
- Signed result provenance: RFC 6962 Merkle trees (domain-separated leaves, no
  duplicated odd leaf), artefact descriptors with inline small outputs and
  hashed external ones, an environment snapshot including whether the tree was
  dirty, and an ML-DSA-signed `.aqresult` bundle.
- `aegisq verify-result`, which reports each check individually and states
  plainly that a valid signature is not evidence of correct computation.
- `docs/security-model.md` and `docs/pqc-protocol.md`: assets, adversaries,
  the order of verification checks and why it matters, the exclusions
  (malicious administrator, memory disclosure, side channels, denial of
  service, key distribution and revocation, metadata privacy), and the
  precise statement of what a signature does and does not prove.
- Shor's algorithm at educational sizes: controlled modular multiplication
  synthesised as a basis-state permutation decomposed into transpositions,
  order finding by phase estimation, and the classical post-processing.
  Factors 15 and 21.
- Algorithm test suite, including a matrix-level comparison of `qft()` against
  the DFT.
- `aegisq benchmark pqc`: ML-KEM and ML-DSA primitive timings and sizes for
  all three security levels, plus the end-to-end cost of packing and opening a
  real job bundle, with the envelope's fixed overhead separated from base64
  expansion of the payload.
- `aegisq benchmark search`: Grover's measured oracle-query scaling against
  classical search, with the measured success probability beside the
  theoretical curve.
- Runnable examples for GHZ, QFT (with a live placement comparison under
  mpirun), Ising, Grover and Shor.
- `docs/limitations.md`: simulation, optimiser, measurement, security and
  algorithm-demonstration constraints in one place.
- Technical report (`paper/main.tex`) whose result tables are generated from
  the processed measurements by `scripts/generate_report.py`, so the paper and
  the README cannot disagree.

### Fixed
- `qft()` iterated the qubits in the wrong direction and produced a circuit
  that was unitary, invertible and *not* the Fourier transform. Phase
  estimation on it was smeared instead of exact. All benchmark measurements
  involving the QFT were re-run after the fix.
