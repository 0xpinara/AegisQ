# Security model

AegisQ-HPC is a **research prototype**. This document states precisely what
the secure job layer protects, what it does not, and why.

## Assets

| Asset | Where it lives | Protection |
|---|---|---|
| circuit | inside the encrypted payload of an `.aqjob` | confidentiality + integrity |
| execution parameters (ranks, precision, shots, seed) | same encrypted payload | confidentiality + integrity |
| client identity | ML-DSA key pair on the submitter's machine | secret half never leaves |
| cluster identity | ML-KEM + ML-DSA key pairs on the cluster | secret half never leaves |
| result counts and metrics | inline artefacts in an `.aqresult` | integrity + origin authentication |
| execution record (environment, placement, timings) | signed result manifest | integrity + origin authentication |

## Adversaries considered

| Adversary | Capability | Outcome |
|---|---|---|
| **Passive observer** | reads bundles in transit or at rest on shared storage | learns routing metadata (job id, timestamps, key fingerprints, ciphertext size); learns nothing about the circuit or its parameters |
| **Active tamperer** | modifies a bundle | any edit to the header breaks AEAD decryption *and* the signature; any edit to the ciphertext or signature breaks the signature. Rejected before execution |
| **Replay attacker** | resubmits a bundle they captured intact | rejected: the job identifier has already been recorded by the cluster |
| **Unauthorised submitter** | has a valid key pair the cluster has never seen | rejected: the signing key is not in the cluster's trusted directory, checked before any decryption is attempted |
| **Result forger** | fabricates or edits a result record | rejected: the manifest is signed with the cluster's ML-DSA key |
| **Downgrade attacker** | rewrites the named algorithms in a bundle | rejected: the crypto suite is inside the signed, AEAD-authenticated header |

### Why the ordering of checks matters

`secure-run` performs, in this order: parse structure → look the signing key up
in the trusted directory → verify the ML-DSA signature → confirm the bundle is
addressed to this cluster → check replay state → decapsulate and decrypt →
re-check the circuit hash → execute.

Nothing inside the envelope is trusted before the signature verifies, so a
malformed or hostile payload never reaches the parser that would interpret it.

## Explicitly out of scope

These are **not** protected against, and no amount of reading the code will
change that:

- **A malicious cluster administrator.** Root on the compute node can read the
  decrypted circuit, alter the computation, and sign whatever result they like
  with the cluster's key.
- **Kernel, hypervisor or firmware compromise** on the submitting or executing
  machine.
- **Memory disclosure during execution.** The circuit and the state vector are
  plaintext in RAM while the job runs. There is no enclave, no memory
  encryption, no confidential computing.
- **Side channels.** Timing, cache, power and traffic-analysis side channels
  are not considered. Communication volume in particular is *deliberately*
  observable — it is the quantity this project measures — and it leaks
  information about circuit structure.
- **Denial of service.** Nothing rate-limits submissions, bounds resource use,
  or prevents a client from filling the replay database.
- **Key distribution.** How a client learns the cluster's genuine public key,
  and how a cluster learns which client keys to trust, is out of band. The
  trusted directory is a local allow-list, not a PKI.
- **Key revocation and rotation.** There is no revocation list and no expiry.
- **Metadata privacy.** Job identifiers, timestamps, key fingerprints, payload
  size and submission patterns are all visible to anyone who can read the file.

## The provenance claim, stated precisely

A valid signature on an `.aqresult` means:

> This record was produced by a holder of the cluster's ML-DSA secret key, and
> has not been modified since it was signed.

It does **not** mean:

> The computation described by this record was performed correctly.

A compromised or simply buggy compute node produces a wrong answer and signs
it with a perfectly valid key. Distinguishing a correct computation from an
incorrect one requires verifiable computation — interactive proofs, succinct
arguments, or replication across independent operators — none of which this
project implements. `aegisq verify-result` prints this distinction in its own
output so a reader cannot take the stronger claim by accident.

## Cryptographic choices

| Purpose | Algorithm | Source | Why |
|---|---|---|---|
| key encapsulation | ML-KEM-768 | liboqs | FIPS 203; NIST level 3, the common "default" parameter set |
| signatures | ML-DSA-65 | liboqs | FIPS 204; NIST level 3, matches the KEM's level |
| key derivation | HKDF-SHA256 | `cryptography` | domain separation between protocol version and job id |
| payload encryption | AES-256-GCM | `cryptography` | authenticated encryption, with the public header as associated data |
| hashing | SHA-256 | `hashlib` | artefact digests and Merkle trees |

No cryptographic primitive is implemented in this repository. Where a
primitive is used, it is used in a standard mode with parameters from the
specification.

### Nonces and randomness

AEAD nonces and replay tokens come from `os.urandom`. A fresh nonce is
generated per job; the AES key is itself unique per job (derived from a fresh
KEM shared secret and bound to the job id), so nonce reuse across jobs is not
a concern even in principle.

## Known weaknesses of this implementation

Stated here rather than discovered later:

- The replay database is a single file per installation. Its `flock` guard is
  dependable on local POSIX storage; on NFS it depends on the mount.
- Pruning the replay database re-enables replay of the pruned identifiers. It
  is opt-in and reports how many entries it dropped.
- `aegisq secure-inspect` intentionally does not verify anything, so operators
  can triage a file before running it. Its output says so on every invocation.
- Timing of the verification path has not been analysed for side channels.
  Signature verification and hash comparison use the underlying libraries'
  implementations without additional constant-time guarantees at this layer.
- There is no audit log beyond the replay database and whatever the scheduler
  records.

## Reporting

See [SECURITY.md](../SECURITY.md). This is a prototype with no deployment
surface; please do not include real key material in a report.
