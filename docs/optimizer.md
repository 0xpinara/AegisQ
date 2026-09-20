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
