# Communication-aware qubit placement

## The problem

Distributing a state vector turns some gates into network operations. *Which*
gates depends entirely on which logical qubits are placed on the `p = log2(P)`
global positions. The default placement (the identity mapping) makes the
highest-numbered qubits global for no better reason than that they are last.

Choosing that set deliberately is the optimisation this project studies.

## Cost model

`aegisq.compiler.cost_model.CommunicationCostModel` predicts, for a circuit
and a candidate set of global qubits, how many bytes will cross the network
and in how many messages. Costs are in bytes rather than arbitrary units so
that predictions can be compared directly against measured counters.

With `S` amplitudes per shard and `w` bytes per amplitude, over `P` ranks:

| Gate | Placement | Bytes (summed over ranks) | Messages |
|---|---|---:|---:|
| `z`, `s`, `t`, `rz`, `cz` | any | 0 | 0 |
| `x`, `y`, `h`, `rx`, `ry` | local qubit | 0 | 0 |
| `x`, `y`, `h`, `rx`, `ry` | global qubit | `P·S·w` | `P` |
| `cx` | local target | 0 | 0 |
| `cx` | local control, global target | `P·S·w/2` | `P` |
| `cx` | global control, global target | `P·S·w/2` | `P/2` |
| `swap` | both local | 0 | 0 |
| `swap` | one global | `P·S·w/2` | `P` |
| `swap` | both global | `P·S·w/2` | `P/2` |

Three consequences matter for the search:

1. **Diagonal structure dominates.** A circuit whose global qubits are only
   touched by `rz`/`cz` costs nothing at all, no matter how deep it is.
2. **Controls are cheap, targets are not.** Placing a qubit that is mostly
   used as a `cx` control on a rank position is nearly free; placing a qubit
   that is mostly a `cx` target is expensive.
3. **Byte volume does not depend on the ordering inside each group.** Only the
   *set* of global qubits matters, which is what makes an exhaustive search
   over `C(n, p)` candidates tractable.

Message count *does* distinguish a global control from a local one (same
bytes, half the messages, twice the size), so it is tracked as a secondary,
latency-flavoured objective.

### Validation

The model is not trusted on its own. `tests/mpi/test_cost_model_prediction.py`
runs each circuit on the real distributed runtime and asserts that predicted
bytes, messages and per-opcode breakdown equal the profiler's measurements —
for random circuits, custom placements, fp32 shards and a placement the model
declares free (which must move zero bytes).

### Two implementations, one rule set

`gate_cost` applies the table above one gate at a time and is the readable,
authoritative version. `CircuitCostProfile` pre-aggregates a circuit so that
thousands of candidate placements can be scored without re-walking the gate
list. A unit test asserts the two agree on randomised circuits, so the fast
path cannot drift away from the rules.

## Search

`aegisq.compiler.static_mapper.StaticCommunicationMapper` selects the set of
`p` global qubits.

| Space | Strategy |
|---|---|
| `C(n, p) <= candidate_budget` (200k by default) | exhaustive — optimal *with respect to the cost model* |
| larger | greedy start, then pairwise swap local search — labelled as not guaranteed optimal |

The greedy start orders qubits by the cost of placing each one globally on its
own, takes the cheapest `p`, and then repeatedly swaps one global qubit for a
local one while that improves the objective.

The objective is lexicographic: **bytes first, message count as a tiebreak**.
Bytes are what placement controls; the message count separates two placements
that move identical volume in different numbers of transfers.

### The result is applied

`MappingResult.mapping` is a logical-to-physical permutation, and it is the
same object the distributed runtime consumes. `tests/mpi/test_optimized_mapping.py`
runs each circuit twice on the live runtime — once with the default placement,
once with the optimised one — and asserts that

1. both produce the same state as the single-process reference, and
2. the measured byte counts equal the predicted ones for both placements, so
   the reported reduction is a reduction in bytes that actually crossed the
   network.

## CLI

```
aegisq optimize random --qubits 20 --ranks 8 --option seed=3
aegisq optimize circuit.json --ranks 4 --json
```

Output states plainly that the figures are predictions; measured numbers come
from a benchmark run.

## A second lever: gate fusion

Placement decides *which* gates communicate. Fusion decides *how many times*.

A single-qubit gate on a global qubit costs a whole-shard exchange. Ten
consecutive such gates cost ten exchanges, even though their product is one
2x2 matrix. `aegisq.compiler.fusion` multiplies each run out before execution,
turning the ten exchanges into one. The arithmetic is unchanged — matrix
multiplication is exactly what the ten separate applications were computing.

Three properties make this safe to apply unconditionally:

1. **It is exact, global phase included.** The fused matrix is carried in full
   (eight real parameters, not Euler angles), so fused and unfused circuits
   can be compared amplitude by amplitude and any discrepancy is a bug rather
   than an expected phase.
2. **It never increases communication.** A run's fused matrix needs at most
   one exchange, where the run needed one per non-diagonal gate. A property
   test asserts this for arbitrary circuits and placements.
3. **It preserves "free".** A run of `rz`, `s` and `z` fuses into a *diagonal*
   matrix, and the runtime and cost model both classify a fused gate by
   inspecting its matrix rather than its opcode — so a diagonal run stays
   communication-free instead of becoming a general unitary.

The fused gate is an internal representation. It has no OpenQASM form (the
subset carries no way to write a raw matrix without losing the global phase),
so the emitter refuses rather than writing something lossy.

### The levers interact

Measured together, the two are not additive. For random circuits, fusion on
its own removes almost nothing, placement removes about a third — and the
combination removes over half. Fusing changes the cost landscape, so the
placement search that runs afterwards finds a different and better assignment.
The measured 2x2 is in the [README](../README.md#measured-results) and in
`benchmarks/processed/lever_comparison.csv`.

## What the model cannot see

The cost model counts bytes and messages. It does not model network topology,
congestion, overlap between computation and communication, NUMA effects or MPI
implementation differences. A predicted reduction in bytes is therefore not a
promise of a proportional reduction in wall-clock time — which is precisely
why the benchmark suite measures both.
