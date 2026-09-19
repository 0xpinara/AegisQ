# Security policy

AegisQ-HPC is a **research prototype**. Do not use it to protect production
workloads or real secrets.

## What the project does

The secure job layer uses NIST-standardised post-quantum primitives from
[liboqs](https://openquantumsafe.org/):

- **ML-KEM-768** (FIPS 203) for key encapsulation,
- **ML-DSA-65** (FIPS 204) for signatures,
- **HKDF-SHA256** for key derivation and **AES-256-GCM** for payload
  encryption, both from the Python `cryptography` package.

No cryptographic primitive is implemented in this repository.

## What the project does not claim

- A signed result manifest authenticates the *origin* of a result and detects
  tampering in transit or at rest. It is **not** a proof that the computation
  was performed correctly.
- The threat model excludes a malicious cluster administrator, kernel or
  hypervisor compromise, memory disclosure on compute nodes, side channels and
  denial of service.

The full model is in [`docs/security-model.md`](docs/security-model.md).

## Reporting an issue

Open a GitHub issue describing the problem. Because this is a prototype with no
deployment surface, there is no private disclosure process; please do not
include real key material in reports.
