# Benchmark methodology

## The rule

Every number in `benchmarks/processed/` and every pixel in
`benchmarks/plots/` is derived from a raw CSV in `benchmarks/raw/` that was
written by an actual run of `aegisq.benchmark.runner`. The plotting code reads
raw data and nothing else: there are no constants describing results anywhere
in the reporting module.

## What a raw row contains

One row per repeat, carrying its own provenance so a measurement is never
separated from the machine that produced it:

| Group | Columns |
|---|---|
| provenance | timestamp, hostname, cpu_model, logical_cores, os, python_version, compiler, mpi_library, git_commit, git_dirty, aegisq_version |
| configuration | experiment, circuit_family, circuit_name, qubits, gates, depth, two_qubit_gates, precision, ranks, omp_threads, thread_policy, mapping_strategy, global_qubits, shots, seed, repeat |
| measurement | wall_seconds, compute_seconds, communication_seconds, bytes_sent, bytes_received, pairwise_exchanges, communicating_gates, local_amplitudes |
| prediction | predicted_bytes, predicted_exchanges |

The prediction columns sit beside the measurement so the cost model can be
audited rather than trusted.

## Statistics

- Each configuration runs a **warm-up repeat that is discarded**: the first
  touch of a fresh shard pays page-fault and allocation costs that say nothing
  about the steady-state cost of the circuit.
- The headline figure is the **minimum** wall time across repeats. On a shared
  machine the least-contended observation is the most reproducible one.
- The median and the spread (max − min) are reported alongside it, so a noisy
  measurement is visible as a noisy measurement.

## Thread policy — and why it is a column

On a single node, how many threads each rank gets changes what the experiment
*means*. The suite records the policy explicitly and never mixes policies in
one table:

| Policy | Setup | Question it answers |
|---|---|---|
| `one-thread-per-rank` | one thread per rank; total cores grow with the rank count | classic strong/weak scaling: does adding processors help? |
| `fixed-total-cores` | `ranks x threads` held at the core count | at constant hardware, what does partitioning cost? |

Reporting a "speedup" from the second policy as if it were the first is one of
the easier ways to publish a misleading scaling curve. Both are measured here
and labelled.

## Definitions

- **Strong scaling**: fixed circuit, growing rank count. Speedup is
  `T(1) / T(P)`; efficiency is `speedup / P`.
- **Weak scaling**: the problem grows with the rank count — one extra qubit per
  doubling — so each rank keeps `2^(n-p)` amplitudes. Ideal behaviour is
  constant wall time.
- **Communication volume**: payload bytes handed to MPI, summed over ranks.
- **Communication time**: seconds spent inside those calls, reported for the
  slowest rank.

## Reproducing a figure

```bash
aegisq benchmark mapping --circuits ghz,qft,ising,random,grover \
    --qubits 20 --ranks 2,4,8 --repeats 3 --option grover:iterations=2
aegisq benchmark strong --circuit qft --qubits 22 --ranks 1,2,4,8 \
    --thread-policy one-thread-per-rank
aegisq benchmark weak --circuit ising --qubits 20 --ranks 1,2,4,8 \
    --thread-policy one-thread-per-rank
aegisq benchmark report
```

Each sweep launches one `mpirun` per configuration, because the world size is
fixed when `mpirun` starts.

## What "measured communication" means

Every byte reported by AegisQ is a byte the runtime handed to MPI. The
counters live in `CommunicationProfiler` and are incremented inside the same
function that issues the transfer, so a code path cannot move data without
being counted.

Counted:

- payload bytes of every `MPI_Sendrecv` issued by the runtime, in both
  directions,
- the number of pairwise exchanges, and which opcode caused each one,
- wall time spent inside those calls,
- collective calls (`MPI_Allreduce` for the norm, `MPI_Allgather` for the
  test-only gather).

Not counted:

- MPI's own protocol and envelope overhead,
- traffic MPI generates internally to implement a collective,
- anything that happens below the MPI interface (NIC, shared memory copies).

Reported byte counts are therefore a lower bound on wire traffic and an exact
account of what the algorithm asked for. That is the quantity the qubit
placement actually controls.

### Local versus world-reduced counters

`metrics()` reports one rank. `reduced_metrics()` reports the world:

- byte counts and call counts are **summed**, because the question is how much
  traffic the job generated in total;
- wall times are **maximised**, because a distributed run finishes when its
  slowest rank finishes.

### Verified byte accounting

The MPI test suite asserts closed-form byte counts for every placement — for
example, a single Hadamard on a global qubit moves exactly one full state
vector's worth of bytes summed over ranks (`2^n · 16` for fp64), and a `CX`
with a local control and a global target moves exactly half of that. These are
assertions, not documentation: an optimisation that changes the traffic
pattern has to change the expectations too.

## Post-quantum measurements

`aegisq benchmark pqc` measures two different things, and the distinction
matters for how the results are read:

- **Primitive cost** — `keygen`, `encapsulate`, `decapsulate`, `sign` and
  `verify` for all three ML-KEM and ML-DSA parameter sets, timed over many
  iterations with the **median** reported. Sizes are exact, not measured.
- **End-to-end cost** — the wall time to pack a real job bundle and to verify,
  decrypt and open it, and the bytes the envelope adds.

The end-to-end figure is much larger than the sum of the primitives, and that
gap is the interesting part: most of it is canonical JSON serialisation and
base64 encoding of the circuit, not lattice arithmetic. The size accounting
separates the three contributions — plaintext payload, base64 expansion, and
the fixed cryptographic overhead (KEM ciphertext, signature, header, AEAD tag)
— so a reader can see which one an optimisation would need to target.

Primitive timings depend on the liboqs build (compiler flags, AVX2/NEON
availability), so the liboqs version is recorded in every row alongside the
CPU.
