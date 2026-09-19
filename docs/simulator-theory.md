# State-vector simulation: conventions and reference implementation

## State representation

An `n`-qubit pure state is a unit vector in a `2^n`-dimensional complex space:

```
|psi> = sum_{i=0}^{2^n - 1} alpha_i |i>,      sum_i |alpha_i|^2 = 1
```

AegisQ stores the amplitudes `alpha_i` explicitly, so memory grows as
`O(2^n)`:

| Precision | Bytes per amplitude | 26 qubits | 30 qubits | 32 qubits | 40 qubits |
|---|---:|---:|---:|---:|---:|
| fp64 (complex128) | 16 | 1 GiB | 16 GiB | 64 GiB | 16 TiB |
| fp32 (complex64) | 8 | 0.5 GiB | 8 GiB | 32 GiB | 8 TiB |

This table is arithmetic, not a measurement: it states what a state vector
costs, which is the reason the runtime must distribute it.

## Index convention

Qubit `q` contributes bit `2^q` to a basis index:

```
i = sum_q b_q 2^q,     b_q = (i >> q) & 1
```

This is the little-endian convention also used by Qiskit, which keeps the
cross-validation suite free of index-reversal bookkeeping.

When a state of `2^n` amplitudes is reshaped to a rank-`n` tensor of shape
`(2,) * n` in C order, tensor axis `a` corresponds to qubit `n - 1 - a`. The
reference backend uses that mapping directly:

```python
axis = num_qubits - 1 - qubit
out  = np.tensordot(U, psi.reshape((2,) * n), axes=([1], [axis]))
psi  = np.moveaxis(out, 0, axis).reshape(-1)
```

Measurement bitstrings are printed with the **highest** measured qubit on the
left, so for a 3-qubit register the key `"100"` means `q2 = 1, q1 = 0, q0 = 0`.

## Supported gate set

| Opcode | Qubits | Params | Diagonal | Notes |
|---|---:|---:|:---:|---|
| `x`, `y`, `z` | 1 | 0 | `z` only | Pauli operators |
| `h` | 1 | 0 | no | Hadamard |
| `s`, `t` | 1 | 0 | yes | `sqrt(Z)` and `Z^(1/4)` |
| `rx`, `ry`, `rz` | 1 | 1 | `rz` only | `exp(-i theta P / 2)` |
| `cx` | 2 | 0 | no | operands are `(control, target)` |
| `cz` | 2 | 0 | yes | symmetric |
| `swap` | 2 | 0 | no | permutation |

The set is intentionally minimal: every opcode must be implemented three times
(reference, C++, distributed) and must have a communication cost rule. Larger
constructions — controlled phase, QFT, Grover oracles, Trotter layers — are
decomposed into these gates.

The `diagonal` column is not cosmetic. A diagonal gate multiplies each
amplitude by a scalar that depends only on that amplitude's basis index, so it
can never move data between MPI ranks. That property is what the
communication cost model in
[`docs/optimizer.md`](optimizer.md) exploits.

### Controlled phase by decomposition

`CP(theta)` is not a primitive. It is expressed as

```
CP(theta) = RZ_c(theta/2) · RZ_t(theta/2) · CX(c,t) · RZ_t(-theta/2) · CX(c,t)
```

up to a global phase (the exact identity uses `P` rather than `RZ`; the two
differ by `exp(i lambda / 2)` per gate). Global phase is unobservable under
measurement, and the cross-validation suite compares states with a global
phase correction for this reason.

## Reference backend

`aegisq.runtime.reference.ReferenceStateVector` is the oracle for every other
backend. It prioritises being obviously correct:

- gates are applied with `tensordot`, one small matrix at a time,
- no in-place tricks, no fused kernels, no threading,
- measurement is terminal and sampled with `numpy.random.Generator.multinomial`
  from a seed, so identical `(circuit, shots, seed)` triples reproduce
  identical counts.

It refuses to allocate more than `2^28` amplitudes unless the caller raises
`max_qubits` explicitly, because an accidental `2^40` allocation is otherwise
the fastest way to freeze a workstation.

## Terminal measurement only

Mid-circuit measurement and classical feed-forward are **not supported**. A
`measure` instruction records that a qubit is sampled after the final gate.
This is a real restriction, not an omission: mid-circuit collapse in a
distributed state vector requires a global normalisation round trip after
every measurement, which would confound the communication measurements that
this project exists to make.
