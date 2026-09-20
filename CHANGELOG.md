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
