# Architecture

AegisQ-HPC is organised as four cooperating layers. Each layer is testable in
isolation, and each one is added by a separate project phase.

```
                    ┌─────────────────────────────┐
                    │        AegisQ client        │
                    │  circuit builder / OpenQASM │
                    │  ML-DSA job signature       │
                    │  ML-KEM session setup       │
                    │  AES-256-GCM encryption     │
                    └──────────────┬──────────────┘
                                   │  encrypted .aqjob
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                           HPC cluster                            │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Secure job loader: ML-DSA verify → replay check →          │  │
│  │ ML-KEM decapsulate → AES-GCM decrypt → circuit hash check  │  │
│  └───────────────────────────┬────────────────────────────────┘  │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Communication-aware compiler                               │  │
│  │ circuit IR → cost model → qubit placement → schedule       │  │
│  └───────────────────────────┬────────────────────────────────┘  │
│                              ▼                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Distributed runtime (C++20 / MPI / OpenMP)                 │  │
│  │ rank 0 shard │ rank 1 shard │ rank 2 shard │ rank 3 shard  │  │
│  │         ↔ instrumented MPI pairwise exchange ↔             │  │
│  └───────────────────────────┬────────────────────────────────┘  │
│                              ▼                                   │
│           metrics + result + Merkle root + ML-DSA signature      │
└──────────────────────────────┬───────────────────────────────────┘
                               ▼
                      aegisq verify-result
```

## Layer responsibilities

| Layer | Module | Responsibility |
|---|---|---|
| Front end | `aegisq.circuit` | gate/circuit IR, validation, OpenQASM subset parsing |
| Reference | `aegisq.runtime.reference` | readable NumPy simulator used as the correctness oracle |
| Native core | `cpp/` | single-process and distributed state-vector kernels |
| Compiler | `aegisq.compiler` | analytical communication cost model and qubit placement |
| Security | `aegisq.secure` | ML-KEM / ML-DSA job envelopes, replay protection |
| Provenance | `aegisq.provenance` | hashing, Merkle trees, signed execution manifests |
| Measurement | `aegisq.benchmark` | scaling sweeps, raw CSV capture, plotting from raw data |

## Design rules

1. The Python layer orchestrates; the C++ layer computes. No amplitude loop
   lives in Python outside the deliberately simple reference backend.
2. Every byte moved by the project over MPI passes through an instrumented
   wrapper, so reported communication volume is measured rather than modelled.
3. Optional components (MPI, OpenMP, liboqs, Qiskit, CUDA) degrade gracefully:
   the package imports and `aegisq doctor` still runs without them.

## Native core layout

| File | Contents |
|---|---|
| `cpp/include/aegisq/gate.hpp` | opcode enum, gate struct, arity/diagonal/control metadata |
| `cpp/include/aegisq/kernels.hpp` | templated local kernels shared by single-process and distributed execution |
| `cpp/include/aegisq/statevector.hpp` | `StateVectorT<Real>`, instantiated for `double` and `float` |
| `cpp/include/aegisq/measurement.hpp` | partition-independent shot sampler |
| `cpp/src/bindings.cpp` | pybind11 surface |

The kernels are templated on the amplitude type and addressed by *local*
qubit index. The distributed runtime hands each rank its own shard and calls
exactly these kernels for every gate that needs no communication, so the two
execution modes share one arithmetic implementation.

### Sampling is independent of the rank count

`measure_all` draws `shots` uniform values in `[0, total_probability)` from a
seeded MT19937-64, sorts them once, and then walks the local amplitudes
accumulating probability mass; a rank claims only the draws inside its own
interval. Because the draw sequence depends on `(shots, seed)` alone, the same
circuit sampled on 1, 2 or 8 ranks yields identical counts.

The reference (NumPy) backend uses `Generator.multinomial` and therefore
produces a *different* — but equally reproducible — stream for the same seed.
Counts are compared across backends statistically; state vectors are compared
exactly.

## Circuit input: a documented OpenQASM subset

`aegisq.circuit.qasm` parses the part of OpenQASM that maps directly onto the
AegisQ instruction set. It is **not** a complete OpenQASM implementation.

Accepted:

| Construct | Forms |
|---|---|
| version header | `OPENQASM 2.0;`, `OPENQASM 3;` (optional) |
| include | `include "qelib1.inc";` (accepted, ignored) |
| quantum register | `qreg q[n];`, `qubit[n] q;` — exactly one per circuit |
| classical register | `creg c[n];`, `bit[n] c;` (accepted, ignored) |
| gates | `x y z h s t` · `rx(theta) ry(theta) rz(theta)` · `cx cz swap` |
| parameters | numbers, `pi`, `pi/4`, `2*pi`, `3*pi/8`, `1e-3` |
| measurement | `measure q[i] -> c[i];`, `c[i] = measure q[i];`, `measure q[i];` |
| barrier | `barrier q;` (accepted, ignored) |
| comments | `//` and `/* ... */` |

Rejected, with a line number and a reason:

- any gate outside the instruction set (including `ccx`/`toffoli`),
- custom `gate` definitions, `if`, loops and any construct with a body,
- `reset`, multiple quantum registers, register-wide gate application,
- unparseable parameter expressions.

Rejecting is the point. A silently ignored instruction would produce a wrong
state and a benchmark that looks perfectly reasonable.

`to_qasm(circuit)` emits the same subset, and a roundtrip test asserts that
parsing the emitted text reproduces the original instruction list and the same
simulated state.
