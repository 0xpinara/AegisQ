# Limitations

Everything below is a real constraint of this implementation, stated so that a
reader does not have to discover it.

## Simulation

- **Memory is the wall.** A state vector costs `2^n` amplitudes. Distribution
  raises the ceiling by the number of nodes, not by an order of magnitude:
  going from 30 to 40 qubits needs a thousand times more memory, not ten.
- **No approximation.** There is no tensor-network, stabiliser or
  matrix-product-state backend. Circuits that those methods handle easily
  (Clifford circuits, low-entanglement dynamics) are simulated here at full
  exponential cost.
- **No noise model.** Everything is noiseless and unitary. There is no
  decoherence, no gate error, no readout error.
- **Terminal measurement only.** Mid-circuit measurement and classical
  feed-forward are rejected rather than approximated, because collapse in a
  distributed state vector requires a global normalisation round trip after
  every measurement, which would confound the communication measurements this
  project exists to make.
- **Twelve gates.** The instruction set is `x y z h s t rx ry rz cx cz swap`.
  Anything else — Toffoli, controlled phase, multi-controlled gates — is
  decomposed, which inflates gate counts relative to hardware-native sets.
- **Power-of-two rank counts.** The partitioning scheme requires `P = 2^p` and
  at least one local qubit. Three ranks is rejected, not silently rounded.
- **fp32 is available but lossy.** Single precision halves memory and wire
  traffic and visibly increases error; cross-backend agreement is asserted at
  `1e-5` rather than `1e-11`.

## The optimiser

- **The cost model counts bytes and messages.** It does not model network
  topology, congestion, overlap of computation with communication, NUMA
  effects or MPI implementation differences. A predicted byte reduction is not
  a promise of a proportional wall-time reduction, and the measurements show
  cases where traffic falls and wall time barely moves.
- **The cost model has no local term.** It treats every local position as
  equivalent. Measured, that is roughly accurate for the kernels that sweep
  the whole state and badly wrong for `cz`, whose achieved bandwidth depends
  strongly on the target qubit's position. The current figures are in
  [`benchmarks/processed/kernel_position_sensitivity.csv`](../benchmarks/processed/kernel_position_sensitivity.csv)
  and summarised in the README rather than repeated here, because a number
  copied into prose stops being true the next time the suite runs. A
  placement chosen purely to minimise network bytes can leave local
  performance on the table.
- **Partial-sweep kernels reach about half the machine's bandwidth.** `cx`,
  `cz` and `swap` plateau near 52–59% of an in-place reference while `h` and
  `rz` reach 98%. This is measured, not modelled, and is the clearest local
  optimisation target in the codebase.
- **Windowed placement uses an approximate transition cost.** Re-mapping
  between circuit segments *is* implemented and measured — it is the third
  lever in the results — but the windows are fixed-width and the dynamic
  program that chooses them scores transitions with an approximation, while
  the rewriter pays the exact cost. The two agreed on the measured circuits
  after a bug that made the planner under-predict its own traffic was fixed;
  they are not guaranteed to agree in general.
- **Exhaustive search is optimal only with respect to the model.** When the
  candidate budget is exceeded the search is a greedy heuristic with local
  improvement, and the result says so.
- **Ordering inside each group is ignored.** Byte volume depends only on which
  qubits are global, so the optimiser does not distinguish placements that
  differ only in local ordering — which could matter for cache behaviour.

## Measurements in this repository

- **One machine.** Every number was measured on a single Apple M2 laptop with
  8 cores and shared-memory MPI. There is no inter-node network involved, so
  the communication times here are best-case: a real cluster interconnect is
  slower, which would make the placement optimisation *more* valuable, not
  less. That is a prediction, not a result.
- **Small problems.** 20-23 qubits, chosen so a sweep finishes in minutes on a
  laptop. The scaling behaviour at 30+ qubits on real nodes is not measured.
- **No GPU results.** The repository has no CUDA backend, and no GPU was
  available on the benchmark host. Nothing here should be read as a statement
  about GPU performance.
- **No external simulator comparison.** Qiskit is used as a correctness oracle
  in the tests, not as a performance baseline. Comparing simulators fairly
  requires care about which optimisations each one applies; that comparison is
  not made here, so no claim is made about relative performance.

## Security

The threat model, its exclusions and the precise meaning of a signed result
are in [`security-model.md`](security-model.md). In brief:

- this is a research prototype, not a production security product;
- a signed result authenticates origin and detects tampering — it is **not**
  evidence that a computation was performed correctly;
- a malicious cluster administrator, memory disclosure during execution and
  side channels are explicitly out of scope;
- the replay database is a single file per installation with `flock`
  semantics that depend on the filesystem.

## Quantum algorithm demonstrations

- **Shor factors 15 and 21.** It is not, and does not approach, a threat to
  RSA. The modular multiplication is synthesised as a basis-state permutation,
  which is general for small moduli and exponential in the register width.
- **Grover searches up to 256 items.** The quadratic query advantage is real
  and measured; the search spaces are toy-sized.
- **No quantum advantage is claimed anywhere.** Every circuit here is
  simulated classically, by definition.
