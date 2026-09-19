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

## Status

Implemented in this phase: layout arithmetic, allocation, reset, globally
reduced norm and a test-only `gather`. Gate execution across ranks is added in
the following phases, in order: local and diagonal gates, global non-diagonal
single-qubit gates, then the four placement cases of `CX`.
