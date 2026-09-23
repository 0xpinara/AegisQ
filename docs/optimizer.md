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

### Why the fallback is rarely a compromise

Look at what the cost rules depend on. A non-diagonal single-qubit gate costs
a full shard if *its* qubit is global. A `cx` costs half a shard if *its
target* is global — whether or not the control is, since the byte volume is
the same either way. Both are functions of one qubit's membership.

`swap` is the exception: it costs if *either* operand is global, and an OR is
not a sum.

So for a circuit containing no `swap` gates the objective is **linear** in the
indicator of the global set, and the optimum is simply the `p` qubits with the
smallest individual costs. The mapper detects this and sorts instead of
searching, reporting the strategy as `linear (separable objective)` and
marking the result optimal — which it is, not approximately.

Where `swap` gates do appear the problem is genuinely combinatorial and the
greedy fallback could in principle settle for a local optimum. Measured, it
does not: across sampled random circuits that all contain swaps, at 4, 8 and
16 ranks, it matched the exhaustive optimum every time while running 3–40x
faster. That is an empirical result on these circuit families, not a proof.

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

## A third lever: windowed placement

Static placement picks one assignment for the whole circuit. That is right for
a circuit whose communication structure is uniform and wrong for one that is
not — a circuit that works intensively on one group of qubits and then moves
to another wants a different assignment in each phase.

`aegisq.compiler.dynamic_mapper` splits the circuit into windows, finds the
best assignment for each, and decides at every boundary:

```
cost(window under the current assignment)
    versus
cost(window under a better one) + cost of getting there
```

Re-assigning is not free. Moving a qubit between a local and a global slot
means moving amplitudes, at half a shard per rank. The plan switches only when
the saving exceeds that.

### Executed as a circuit rewrite

A plan is applied by rewriting the circuit, not by adding a runtime feature:
gates are re-expressed on slots, a change of assignment becomes SWAP gates
between slots, and the resulting traffic is measured by the same profiler as
everything else. The runtime keeps one fixed mapping and needs no knowledge of
the plan.

The final permutation is left in place and the *results* are relabelled, which
is free classical bookkeeping — exactly what the runtime mapping does for a
static assignment. Making the circuit restore logical order instead is
available (`restore_order=True`) and costs real exchanges; charging the
windowed plan for something the static baseline gets for free would have made
the comparison meaningless. That asymmetry was in the first version of this
module and showed up as windowing "losing" to static placement on circuits
where it plainly should not.

### The planner and the rewriter must agree

The search uses a transition estimate that depends only on the two
assignments, because a path-dependent cost cannot be optimised over. The cost
finally *reported* is recomputed from the exact swap list the rewriter will
emit. An earlier version reported the estimate, which missed the
global-to-global swaps a restoration needs and under-predicted its own traffic
— caught by comparing prediction against measurement on the live runtime,
which is the check every part of this project is held to.

### Measured

Windowed placement beats the best single assignment on four of the five
benchmark families; for the QFT it removes 95.7% of baseline traffic against
87.2% for the best static assignment. On GHZ, which has no phase structure,
the planner declines to switch and the two coincide. The numbers are in the
[README](../README.md#measured-results) and
`benchmarks/processed/lever_comparison.csv`.

## What the model cannot see

The cost model counts bytes and messages. It does not model network topology,
congestion, overlap between computation and communication, NUMA effects or MPI
implementation differences. A predicted reduction in bytes is therefore not a
promise of a proportional reduction in wall-clock time — which is precisely
why the benchmark suite measures both.

## Ordering the local qubits

The cost model decides which qubits are global. Everything below that cut is
interchangeable as far as traffic goes: two placements with the same global
set send byte for byte the same data. The optimiser therefore had nothing to
say about local order and left it at the identity.

The kernel sweep says the order is not free in time. At 22 qubits on eight
threads a `cz` on local position 1 runs at 22 GB/s and the same gate on
position 21 runs at 47. `h` and `rz` lean the other way by rather less. So
there is a choice to make, and it costs no traffic to make it.

`optimise_local_order` counts how often each qubit is the position-sensitive
operand of each opcode — the *target* for two-qubit gates, since that is what
the sweep varies — prices every (qubit, position) pair by interpolating the
measured table, and solves the resulting assignment problem. The objective is
a sum of per-qubit terms, so this is a linear assignment and the Hungarian
algorithm gives the exact optimum. It is written out in `local_order.py`
rather than taken from scipy, which is not a dependency of the package.

What it is worth, on the circuits measured here:

| circuit | predicted local-time saving |
|---|---:|
| qft | 0.4% |
| ising | 0.0% |
| grover | 1.6% |
| random | 2.9% |
| a synthetic `cz`-heavy circuit | 19–23% |

The honest reading is that the freedom exists, the algorithm exploits it
exactly, and for this benchmark mix there is almost nothing to exploit — the
circuits are dominated by `rz`, `cx` and `h`, whose position spread is small,
and `cz` is rare in all of them. The predicted savings sit below the noise
floor measured in [benchmark-methodology.md](benchmark-methodology.md), so
they are not reported as speedups. The synthetic case is there to show the
mechanism does what the kernel data says it should when the gate mix calls
for it.

Two assumptions are worth knowing. Costs between the three measured positions
are interpolated linearly, and three points cannot tell you whether the curve
is straight. And the table is measured at one circuit width and applied at
all of them; on a circuit much narrower than 22 qubits every local position
falls at or below the first anchor, so the model has nothing to say rather
than something wrong to say.
