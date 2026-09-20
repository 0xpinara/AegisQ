# The AegisQ post-quantum job protocol

## Overview

```
client                                            cluster
------                                            -------
circuit + parameters
        │
        ├─ ML-KEM-768 encapsulate to cluster public key ──► (ct, ss)
        ├─ HKDF-SHA256(ss, "aegisq/pqc/v1|job|<job_id>") ──► AES key
        ├─ AES-256-GCM(payload, aad = header)          ──► ciphertext
        └─ ML-DSA-65 sign({header, ciphertext})        ──► signature
                                   │
                              .aqjob file
                                   │
                                   ▼
                        1. parse structure
                        2. client key ∈ trusted set?
                        3. verify ML-DSA signature
                        4. addressed to this cluster?
                        5. job id unseen?  (replay state)
                        6. ML-KEM decapsulate → HKDF → AES key
                        7. AES-256-GCM decrypt (aad = header)
                        8. circuit SHA-256 matches manifest?
                        9. execute
                                   │
                        ML-DSA sign result manifest
                                   │
                            .aqresult file
```

## Bundle layout

```json
{
  "schema": "aegisq.job.envelope.v1",
  "protected": {
    "header": {
      "schema": "aegisq.job.header.v1",
      "job_id": "...",
      "created_at": "...",
      "crypto_suite": {"kem": "ML-KEM-768", "signature": "ML-DSA-65",
                       "kdf": "HKDF-SHA256", "aead": "AES-256-GCM"},
      "client_fingerprint": "...", "client_name": "...",
      "cluster_fingerprint": "...", "cluster_name": "...",
      "kem_ciphertext": "<base64>",
      "aead_nonce": "<base64>",
      "payload_bytes": 1234
    },
    "ciphertext": "<base64>"
  },
  "signature": "<base64>"
}
```

The **header is the AEAD associated data** and is also covered by the
signature. That double binding is deliberate:

- editing a header field (say, `ranks` inside the manifest's addressee, or the
  `cluster_fingerprint`) breaks decryption, so a job can never run under
  parameters other than the ones that were signed;
- editing the ciphertext breaks the signature;
- swapping a header from one bundle onto the ciphertext of another breaks both.

## Canonical serialisation

Everything that is signed or hashed is serialised as UTF-8 JSON with sorted
keys, no insignificant whitespace, and no NaN or Infinity. The parser
**re-serialises what it parsed and compares bytes**, rejecting anything that
was not already canonical. Without that check, an attacker could present a
semantically identical but byte-different document, and a naive verifier
reconstructing the "canonical" form might verify a signature over bytes that
differ from what it actually parsed.

## Encrypted payload

```json
{
  "schema": "aegisq.job.payload.v1",
  "manifest": {
    "schema": "aegisq.job.manifest.v1",
    "job_id": "...", "created_at": "...", "nonce": "<base64>",
    "circuit": {"name": "...", "sha256": "...", "num_qubits": 24,
                "gates": 1250, "depth": 300},
    "execution": {"ranks": 8, "precision": "fp64", "shots": 1024,
                  "seed": 42, "mapping_strategy": "optimized"},
    "software": {"aegisq_version": "0.1.0"}
  },
  "circuit": { "schema": "aegisq.circuit.v1", "...": "..." }
}
```

The manifest and the circuit are encrypted **together**, so execution
parameters can never be separated from the program they describe. The circuit
hash inside the manifest is re-checked after decryption, which catches a
mismatch that somehow survived both the AEAD and the signature.

## Result bundle

```json
{
  "schema": "aegisq.result.envelope.v1",
  "manifest": {
    "schema": "aegisq.result.manifest.v1",
    "job_id": "...", "created_at": "...",
    "client_fingerprint": "...",
    "input": {"circuit_sha256": "..."},
    "artifacts": [{"name": "counts.json", "size_bytes": 812,
                   "sha256": "...", "chunk_size": 1048576,
                   "chunk_count": 1, "merkle_root": "...",
                   "inline": "<base64>"}],
    "output_merkle_root": "...",
    "execution": {"world_size": 8, "precision": "fp64", "...": "..."},
    "performance": {"wall_seconds": 12.3, "compute_seconds": 9.8},
    "communication": {"bytes_sent": 12884901888, "pairwise_exchanges": 4096},
    "environment": {"git_commit": "...", "git_dirty": false, "...": "..."}
  },
  "cluster_name": "...", "cluster_fingerprint": "...",
  "signature": "<base64>"
}
```

Artefacts small enough to inline travel inside the bundle; larger ones are
referenced by name and verified against files on disk when they are available.
The Merkle root over the artefact list commits to their **order** as well as
their contents.

## Replay protection

The cluster records every accepted job identifier. A resubmitted bundle — even
a perfectly valid one — is refused. Under `mpirun`, every rank consults the
replay state and refuses independently, while only rank 0 records the
acceptance; that way the ranks reach the same verdict without needing to
exchange it, and a rejection cannot leave some ranks waiting at a barrier.

## Overhead

The cost of the security layer is measured, not estimated: see
`aegisq benchmark pqc` and the results in
[`docs/benchmark-methodology.md`](benchmark-methodology.md). For orientation,
the fixed sizes are:

| Item | Bytes |
|---|---:|
| ML-KEM-768 public key | 1184 |
| ML-KEM-768 ciphertext | 1088 |
| ML-KEM-768 shared secret | 32 |
| ML-DSA-65 public key | 1952 |
| ML-DSA-65 signature | 3309 |

An envelope therefore adds roughly 4.5 KiB of fixed overhead regardless of
circuit size — negligible beside a job that will move gigabytes over MPI, and
the reason the protocol is file-oriented rather than streaming.

## What this protocol does not do

See [`security-model.md`](security-model.md). In particular: it does not
protect against a malicious cluster operator, and a signed result is not proof
of correct computation.
