# Distributed state-vector partitioning

## Partitioning scheme

With `P = 2^p` MPI ranks and `n` qubits, each rank stores

```
2^L amplitudes,    L = n - p
```

A *physical* basis index is split into a rank part and a local part:

```
physical_index = (rank << L) | local_index

rank        = physical_index >> L
local_index = physical_index & ((1 << L) - 1)
```

The low `L` bit positions are **local**: both values of such a bit live inside
one rank's shard. The high `p` positions are **global**: flipping one of them
moves to a different rank.

The current implementation requires `P` to be a power of two and `L >= 1`
(at least one local qubit). Both are checked in `DistributedLayout`.

## Placement is a free choice

Nothing forces logical qubit `q` to occupy physical position `q`.
`DistributedLayout` carries an explicit permutation

```
logical_to_position : [0, n) -> [0, n)
```

that defaults to the identity — which is why the *highest-numbered* qubits are
global by default. The communication-aware mapper (Phase 10) replaces this
permutation; the runtime and the cost model read placement only through
`is_local`, `is_global`, `global_position` and `partner_rank_for_global_qubit`,
so no kernel needs to know that a mapping was applied.

`gather()` translates back to logical basis order before returning, so tests
and callers compare against the single-process reference without knowing the
placement.

## Communication classes

The cost of a gate on a global qubit depends on the gate's structure, not just
on the fact that the qubit is global.

| Gate | Operand placement | Communication |
|---|---|---|
| `z`, `s`, `t`, `rz` | local | none |
| `z`, `s`, `t`, `rz` | global | **none** — the rank id fixes the bit value, so the whole shard is scaled by one factor |
| `x`, `y`, `h`, `rx`, `ry` | local | none |
| `x`, `y`, `h`, `rx`, `ry` | global | pairwise exchange with `rank ^ (1 << global_position)` |
| `cz` | any | **none** — diagonal in both operands |
| `cx` | control local, target local | none |
| `cx` | control global, target local | **none** — ranks whose control bit is 1 apply `X` locally |
| `cx` | control local, target global | pairwise exchange |
| `cx` | control global, target global | exchange, but only between rank pairs whose control bit is 1 |
| `swap` | both local | none |
| `swap` | one or both global | exchange |

Two entries in that table are the reason this project exists: "global" does
not mean "expensive", and a control qubit is cheaper to place globally than a
target qubit. A placement heuristic that simply avoids global qubits, or that
treats all global qubits equally, leaves measurable bandwidth on the table.

## Partner ranks

For a gate on global qubit `q` with global position `g = position(q) - L`:

```
partner_rank = rank XOR (1 << g)
```

The relation is an involution, so the two ranks of a pair agree on who they
are exchanging with without any negotiation.

## Initial state

`|0...0>` is physical index 0 under every permutation, and index 0 lives on
rank 0. Initialisation therefore requires no communication: rank 0 writes a
single amplitude, every other rank zeroes its shard.

## How much data actually moves

Not every communicating gate costs the same. With `S = 2^L` amplitudes per
rank and `w` bytes per amplitude:

| Gate and placement | Bytes sent per participating rank | Participating ranks |
|---|---:|---|
| `x`/`y`/`h`/`rx`/`ry` on a global qubit | `S·w` | all |
| `cx` (local control, global target) | `S·w / 2` | all |
| `cx` (global control, global target) | `S·w` | half (those with the control bit set) |
| `swap` (local, global) | `S·w / 2` | all |
| `swap` (global, global) | `S·w` | half (those whose two bits differ) |
| everything in the zero-communication table above | 0 | none |

The halved cases are not an approximation. In a `CX` with a local control, only
the amplitudes whose control bit is set move; those amplitudes form contiguous
runs of length `2^position(control)`, so they are packed into a contiguous
buffer with a strided block copy, exchanged, and scattered back. (An MPI
derived datatype could describe the same strided access without packing; an
explicit pack was chosen because it keeps the measured byte count
unambiguous.)

### Why a global control is cheap

A control qubit is only read. When it sits on a global position its value is a
property of the rank id, so ranks that do not satisfy it skip the gate
entirely and ranks that do satisfy it proceed with a purely local operation
(`CX` with a local target) or a shard swap (`CX` with a global target). A
*target* qubit, by contrast, is written, and writing a global bit means moving
amplitudes.

This asymmetry — controls are cheap globally, targets are expensive — is the
structure the communication-aware mapper exploits.

## Exchange protocol

All pairwise traffic uses a symmetric `MPI_Sendrecv` between a rank and its
partner, which cannot deadlock: both sides post the same call with the same
counts. Incoming data lands in a separate buffer, so no amplitude is
overwritten before it has been consumed. Buffers are allocated once and
reused, and transfers are chunked at `2^28` elements because MPI element
counts are `int`-typed.

## Status

Implemented: layout arithmetic, allocation, reset, allreduced norm, test-only
`gather`, and every gate in the supported set across all placements. The
remaining phases add instrumentation (measured bytes and time), an analytical
cost model, and the mapper that chooses the placement.

## Which calls are collective

Four methods on a distributed state reduce or gather across the whole world,
and every rank has to make the call:

| Call | Underlying | Why |
|---|---|---|
| `norm()` | `MPI_Allreduce` | the norm is a sum over all shards |
| `gather()` | `MPI_Allgather` | assembles the whole vector on every rank |
| `measure_all(shots, seed)` | `MPI_Allreduce` + `MPI_Allgatherv` | the sampler walks probability mass that lives on every rank |
| `reduced_metrics()` | `MPI_Allreduce` | sums byte counters and takes the slowest time |

Skipping one on a single rank does not raise anything. The ranks that did call
it block inside the reduction, the rank that skipped it runs ahead, and the job
stops with no output and no error — which reads exactly like a hang in the
simulation itself. The usual way to write the bug is:

```python
if rank() == 0:
    print(state.norm())        # every other rank is now stuck
```

The fix is to call it everywhere and print in one place:

```python
value = state.norm()           # all ranks
if rank() == 0:
    print(value)
```

`local_squared_norm()` is the non-collective counterpart, and is what to reach
for when a per-rank number is all that is needed.
