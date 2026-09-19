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
