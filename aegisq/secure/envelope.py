"""Authenticated, encrypted job bundles.

A job travels as a single file (`.aqjob`) so it can move over whatever a
cluster already has — scp, a shared filesystem, a Slurm staging directory —
without AegisQ needing to run a network service.

Structure
---------

```
envelope
├── schema, signature            ← outer wrapper
└── protected                    ← everything the signature covers
    ├── header                   ← public metadata, also the AEAD associated data
    │   ├── schema, job_id, created_at
    │   ├── crypto_suite         ← named algorithms, not "trust me"
    │   ├── client_fingerprint   ← who signed
    │   ├── cluster_fingerprint  ← who can open it
    │   ├── kem_ciphertext       ← ML-KEM encapsulation to the cluster
    │   └── aead_nonce
    └── ciphertext               ← AES-256-GCM over {manifest, circuit}
```

Binding, and why it is arranged this way:

* the **header is the AEAD associated data**, so editing any public field —
  the job id, the rank count, the cluster fingerprint — makes decryption fail
  rather than silently producing a job that runs under different parameters;
* the **signature covers the header *and* the ciphertext**, so an attacker who
  swaps in a different payload invalidates the signature;
* the signature is verified **before** anything inside the envelope is trusted,
  and the manifest is only parsed after decryption succeeds.

The encrypted payload holds the manifest and the circuit together, so the
parameters can never be separated from the program they describe.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aegisq import __version__
from aegisq.circuit.circuit import Circuit
from aegisq.secure import kem
from aegisq.secure.canonical import (
    CanonicalisationError,
    b64decode,
    b64encode,
    canonical_bytes,
    canonical_hash,
    parse_canonical,
    sha256_hex,
)
from aegisq.secure.keys import (
    KEM_ALGORITHM,
    SIGNATURE_ALGORITHM,
    Identity,
    IdentityError,
    PublicIdentity,
)
from aegisq.secure.signatures import SignatureError, sign_payload, verify_payload

ENVELOPE_SCHEMA = "aegisq.job.envelope.v1"
HEADER_SCHEMA = "aegisq.job.header.v1"
MANIFEST_SCHEMA = "aegisq.job.manifest.v1"
PAYLOAD_SCHEMA = "aegisq.job.payload.v1"

#: Named algorithms travel with the envelope so a verifier never has to guess.
CRYPTO_SUITE = {
    "kem": KEM_ALGORITHM,
    "signature": SIGNATURE_ALGORITHM,
    "kdf": "HKDF-SHA256",
    "aead": "AES-256-GCM",
}


class EnvelopeError(RuntimeError):
    """Raised when a bundle is malformed, unauthorised or fails verification."""


@dataclass(frozen=True)
class ExecutionRequest:
    """What the client is asking the cluster to do."""

    ranks: int = 1
    precision: str = "fp64"
    shots: int = 1024
    seed: int = 42
    mapping_strategy: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranks": int(self.ranks),
            "precision": str(self.precision),
            "shots": int(self.shots),
            "seed": int(self.seed),
            "mapping_strategy": str(self.mapping_strategy),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExecutionRequest:
        return cls(
            ranks=int(payload["ranks"]),
            precision=str(payload["precision"]),
            shots=int(payload["shots"]),
            seed=int(payload["seed"]),
            mapping_strategy=str(payload.get("mapping_strategy", "default")),
        )


@dataclass(frozen=True)
class JobManifest:
    """The protected description of a job."""

    job_id: str
    created_at: str
    nonce: str
    circuit_sha256: str
    circuit_name: str
    num_qubits: int
    gate_count: int
    depth: int
    execution: ExecutionRequest
    software: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MANIFEST_SCHEMA,
            "job_id": self.job_id,
            "created_at": self.created_at,
            "nonce": self.nonce,
            "circuit": {
                "name": self.circuit_name,
                "sha256": self.circuit_sha256,
                "num_qubits": self.num_qubits,
                "gates": self.gate_count,
                "depth": self.depth,
            },
            "execution": self.execution.to_dict(),
            "software": self.software,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> JobManifest:
        if payload.get("schema") != MANIFEST_SCHEMA:
            raise EnvelopeError(f"unknown job manifest schema {payload.get('schema')!r}")
        circuit = payload["circuit"]
        return cls(
            job_id=str(payload["job_id"]),
            created_at=str(payload["created_at"]),
            nonce=str(payload["nonce"]),
            circuit_sha256=str(circuit["sha256"]),
            circuit_name=str(circuit["name"]),
            num_qubits=int(circuit["num_qubits"]),
            gate_count=int(circuit["gates"]),
            depth=int(circuit["depth"]),
            execution=ExecutionRequest.from_dict(payload["execution"]),
            software=dict(payload.get("software", {})),
        )


@dataclass(frozen=True)
class OpenedJob:
    """A verified, decrypted job, ready to execute."""

    manifest: JobManifest
    circuit: Circuit
    client: PublicIdentity
    header: dict[str, Any]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def circuit_digest(circuit: Circuit) -> str:
    """SHA-256 over the circuit's canonical dictionary form."""
    return canonical_hash(circuit.to_dict())


def build_manifest(
    circuit: Circuit,
    execution: ExecutionRequest,
    job_id: str | None = None,
    software: dict[str, Any] | None = None,
) -> JobManifest:
    """Describe a job, including a fresh replay nonce."""
    import os

    return JobManifest(
        job_id=job_id or uuid.uuid4().hex,
        created_at=_now(),
        nonce=b64encode(os.urandom(16)),
        circuit_sha256=circuit_digest(circuit),
        circuit_name=circuit.name,
        num_qubits=circuit.num_qubits,
        gate_count=len(circuit),
        depth=circuit.depth(),
        execution=execution,
        software=software or {"aegisq_version": __version__},
    )


def pack_job(
    circuit: Circuit,
    execution: ExecutionRequest,
    client: Identity,
    cluster_public: PublicIdentity,
    job_id: str | None = None,
    software: dict[str, Any] | None = None,
) -> bytes:
    """Produce the bytes of a signed, encrypted `.aqjob` bundle."""
    if cluster_public.kem_public_key is None:
        raise EnvelopeError(
            f"{cluster_public.name} has no KEM key; a job can only be sent to a cluster identity"
        )
    if client.public.role != "client":
        raise EnvelopeError(f"{client.public.name} is not a client identity")

    manifest = build_manifest(circuit, execution, job_id=job_id, software=software)
    payload = {
        "schema": PAYLOAD_SCHEMA,
        "manifest": manifest.to_dict(),
        "circuit": circuit.to_dict(),
    }
    plaintext = canonical_bytes(payload)

    encapsulation = kem.encapsulate(cluster_public.kem_public_key, cluster_public.kem_algorithm)
    key = kem.session_key_for_job(encapsulation.shared_secret, manifest.job_id)
    nonce = kem.random_nonce()

    header = {
        "schema": HEADER_SCHEMA,
        "job_id": manifest.job_id,
        "created_at": manifest.created_at,
        "crypto_suite": dict(CRYPTO_SUITE),
        "client_fingerprint": client.public.signature_fingerprint,
        "client_name": client.public.name,
        "cluster_fingerprint": cluster_public.kem_fingerprint,
        "cluster_name": cluster_public.name,
        "kem_ciphertext": b64encode(encapsulation.ciphertext),
        "aead_nonce": b64encode(nonce),
        "payload_bytes": len(plaintext),
    }

    # The header is the associated data: editing any public field breaks
    # decryption, not just the signature.
    _, ciphertext = kem.encrypt(key, plaintext, canonical_bytes(header), nonce=nonce)

    protected = {"header": header, "ciphertext": b64encode(ciphertext)}
    signature = sign_payload(client.secret, protected)

    return canonical_bytes(
        {
            "schema": ENVELOPE_SCHEMA,
            "protected": protected,
            "signature": b64encode(signature),
        }
    )


def _parse_envelope(raw: bytes) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    try:
        envelope = parse_canonical(raw)
    except CanonicalisationError as exc:
        raise EnvelopeError(f"job bundle is not a canonical AegisQ envelope: {exc}") from None
    if not isinstance(envelope, dict) or envelope.get("schema") != ENVELOPE_SCHEMA:
        raise EnvelopeError(f"unknown envelope schema {envelope.get('schema')!r}")

    protected = envelope.get("protected")
    signature = envelope.get("signature")
    if not isinstance(protected, dict) or not isinstance(signature, str):
        raise EnvelopeError("envelope is missing its protected section or signature")
    header = protected.get("header")
    if not isinstance(header, dict) or header.get("schema") != HEADER_SCHEMA:
        raise EnvelopeError("envelope header is missing or has an unknown schema")
    return protected, header, b64decode(signature)


def inspect_job(raw: bytes) -> dict[str, Any]:
    """Public metadata of a bundle, without decrypting anything.

    Deliberately does **not** verify the signature: this is what an operator
    runs to see where a file came from before deciding what to do with it.
    The returned dictionary says so explicitly.
    """
    protected, header, signature = _parse_envelope(raw)
    ciphertext = b64decode(str(protected["ciphertext"]))
    return {
        "job_id": header.get("job_id"),
        "created_at": header.get("created_at"),
        "crypto_suite": header.get("crypto_suite"),
        "client_name": header.get("client_name"),
        "client_fingerprint": header.get("client_fingerprint"),
        "cluster_name": header.get("cluster_name"),
        "cluster_fingerprint": header.get("cluster_fingerprint"),
        "ciphertext_bytes": len(ciphertext),
        "signature_bytes": len(signature),
        "envelope_sha256": sha256_hex(raw),
        "verified": False,
        "note": "metadata only; the signature has not been checked and the payload is encrypted",
    }


def open_job(
    raw: bytes,
    cluster: Identity,
    trusted_clients: dict[str, PublicIdentity],
) -> OpenedJob:
    """Verify and decrypt a bundle.

    The order matters and is enforced here:

    1. parse the envelope structurally,
    2. resolve the signing key against the cluster's allow-list — an unknown
       client is rejected before any cryptography is spent on it,
    3. verify the signature over the protected section,
    4. check the envelope was addressed to *this* cluster,
    5. decapsulate, derive, decrypt,
    6. only then parse the manifest and circuit, and check the circuit hash.
    """
    protected, header, signature = _parse_envelope(raw)

    fingerprint = str(header.get("client_fingerprint", ""))
    client = trusted_clients.get(fingerprint)
    if client is None:
        raise EnvelopeError(
            f"client key {fingerprint or '<missing>'} is not in the cluster's trusted set"
        )

    if not verify_payload(client, protected, signature):
        raise SignatureError(
            f"job envelope signature does not verify against {client.name} ({fingerprint})"
        )

    suite = header.get("crypto_suite") or {}
    if suite.get("kem") != CRYPTO_SUITE["kem"] or suite.get("aead") != CRYPTO_SUITE["aead"]:
        raise EnvelopeError(f"unsupported crypto suite in envelope: {suite}")

    if cluster.secret.kem_secret_key is None or cluster.public.kem_fingerprint is None:
        raise IdentityError(f"{cluster.public.name} has no KEM key and cannot open jobs")
    if header.get("cluster_fingerprint") != cluster.public.kem_fingerprint:
        raise EnvelopeError(
            "job is addressed to a different cluster key "
            f"({header.get('cluster_fingerprint')}, this cluster is "
            f"{cluster.public.kem_fingerprint})"
        )

    shared_secret = kem.decapsulate(
        b64decode(str(header["kem_ciphertext"])),
        cluster.secret.kem_secret_key,
        cluster.secret.kem_algorithm or KEM_ALGORITHM,
    )
    key = kem.session_key_for_job(shared_secret, str(header["job_id"]))
    plaintext = kem.decrypt(
        key,
        b64decode(str(header["aead_nonce"])),
        b64decode(str(protected["ciphertext"])),
        canonical_bytes(header),
    )

    try:
        payload = parse_canonical(plaintext)
    except CanonicalisationError as exc:
        raise EnvelopeError(f"decrypted payload is not canonical: {exc}") from None
    if payload.get("schema") != PAYLOAD_SCHEMA:
        raise EnvelopeError(f"unknown payload schema {payload.get('schema')!r}")

    manifest = JobManifest.from_dict(payload["manifest"])
    circuit = Circuit.from_dict(payload["circuit"])

    if manifest.job_id != header.get("job_id"):
        raise EnvelopeError("job id inside the envelope does not match the header")
    if circuit_digest(circuit) != manifest.circuit_sha256:
        raise EnvelopeError("circuit does not match the hash recorded in the manifest")
    if circuit.num_qubits != manifest.num_qubits or len(circuit) != manifest.gate_count:
        raise EnvelopeError("circuit shape does not match the manifest")

    return OpenedJob(manifest=manifest, circuit=circuit, client=client, header=header)


def write_job(path: Path, raw: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def read_job(path: Path) -> bytes:
    path = Path(path)
    if not path.exists():
        raise EnvelopeError(f"job bundle not found: {path}")
    return path.read_bytes()
