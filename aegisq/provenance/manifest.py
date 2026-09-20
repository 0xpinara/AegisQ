"""Signed execution records.

A result manifest states what was run, on what, by whom, and what came out —
and is signed by the cluster's ML-DSA identity so that any later modification
is detectable.

What this does and does not mean
--------------------------------
A valid signature proves the record was produced by the holder of the
cluster's signing key and has not been altered since. It does **not** prove
the computation was performed correctly. A compromised compute node can sign
a wrong answer with a perfectly valid key. Verifying provenance is not
verifying computation; see `docs/security-model.md`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aegisq.provenance.environment import capture as capture_environment
from aegisq.provenance.hashing import Artifact, check_artifact, describe_bytes
from aegisq.provenance.merkle import merkle_root
from aegisq.secure.canonical import (
    CanonicalisationError,
    b64decode,
    b64encode,
    canonical_bytes,
    parse_canonical,
)
from aegisq.secure.keys import Identity, PublicIdentity
from aegisq.secure.signatures import sign_payload, verify_payload

RESULT_SCHEMA = "aegisq.result.manifest.v1"
RESULT_ENVELOPE_SCHEMA = "aegisq.result.envelope.v1"


class ProvenanceError(RuntimeError):
    """Raised when a result bundle is malformed or fails verification."""


@dataclass(frozen=True)
class ResultManifest:
    """Everything needed to audit one execution."""

    job_id: str
    created_at: str
    input_circuit_sha256: str
    artifacts: list[Artifact]
    output_merkle_root: str
    execution: dict[str, Any]
    performance: dict[str, Any]
    communication: dict[str, Any]
    environment: dict[str, Any] = field(default_factory=dict)
    client_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RESULT_SCHEMA,
            "job_id": self.job_id,
            "created_at": self.created_at,
            "client_fingerprint": self.client_fingerprint,
            "input": {"circuit_sha256": self.input_circuit_sha256},
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "output_merkle_root": self.output_merkle_root,
            "execution": self.execution,
            "performance": self.performance,
            "communication": self.communication,
            "environment": self.environment,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ResultManifest:
        if payload.get("schema") != RESULT_SCHEMA:
            raise ProvenanceError(f"unknown result manifest schema {payload.get('schema')!r}")
        return cls(
            job_id=str(payload["job_id"]),
            created_at=str(payload["created_at"]),
            input_circuit_sha256=str(payload["input"]["circuit_sha256"]),
            artifacts=[Artifact.from_dict(item) for item in payload["artifacts"]],
            output_merkle_root=str(payload["output_merkle_root"]),
            execution=dict(payload.get("execution", {})),
            performance=dict(payload.get("performance", {})),
            communication=dict(payload.get("communication", {})),
            environment=dict(payload.get("environment", {})),
            client_fingerprint=payload.get("client_fingerprint"),
        )


def compute_output_root(artifacts: list[Artifact]) -> str:
    """Merkle root over the artefacts' own roots, in listed order.

    Order is part of the commitment: re-ordering the artefact list changes the
    root, so a bundle cannot be rearranged without breaking verification.
    """
    leaves = [bytes.fromhex(artifact.merkle_root) for artifact in artifacts]
    return merkle_root(leaves).hex()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_result_manifest(
    job_id: str,
    input_circuit_sha256: str,
    artifacts: list[Artifact],
    execution: dict[str, Any],
    performance: dict[str, Any],
    communication: dict[str, Any],
    client_fingerprint: str | None = None,
    environment: dict[str, Any] | None = None,
) -> ResultManifest:
    return ResultManifest(
        job_id=job_id,
        created_at=_now(),
        input_circuit_sha256=input_circuit_sha256,
        artifacts=artifacts,
        output_merkle_root=compute_output_root(artifacts),
        execution=execution,
        performance=performance,
        communication=communication,
        environment=environment or capture_environment(),
        client_fingerprint=client_fingerprint,
    )


def artifacts_from_result(
    counts: dict[str, int],
    metrics: dict[str, Any],
) -> list[Artifact]:
    """Standard artefacts for a simulation result: counts and metrics."""
    return [
        describe_bytes("counts.json", canonical_bytes(counts)),
        describe_bytes("metrics.json", canonical_bytes(metrics)),
    ]


def sign_result(manifest: ResultManifest, cluster: Identity) -> bytes:
    """Produce the bytes of a signed `.aqresult` bundle."""
    payload = manifest.to_dict()
    signature = sign_payload(cluster.secret, payload)
    return canonical_bytes(
        {
            "schema": RESULT_ENVELOPE_SCHEMA,
            "manifest": payload,
            "cluster_name": cluster.public.name,
            "cluster_fingerprint": cluster.public.signature_fingerprint,
            "signature": b64encode(signature),
        }
    )


@dataclass(frozen=True)
class VerificationReport:
    """The outcome of checking a result bundle, item by item."""

    ok: bool
    job_id: str | None
    cluster_name: str | None
    cluster_fingerprint: str | None
    checks: list[tuple[str, bool, str]]
    manifest: ResultManifest | None = None

    def summary(self) -> str:
        lines = [f"Result bundle for job {self.job_id or '<unknown>'}"]
        if self.cluster_name:
            lines.append(f"  signed by: {self.cluster_name} ({self.cluster_fingerprint})")
        lines.append("")
        for name, ok, detail in self.checks:
            lines.append(f"  [{'ok' if ok else 'FAIL'}] {name}: {detail}")
        lines.append("")
        lines.append("  VERIFIED" if self.ok else "  VERIFICATION FAILED")
        if self.ok:
            lines.append("")
            lines.append(
                "  A valid signature authenticates the origin of this record and detects tampering."
            )
            lines.append(
                "  It is not evidence that the computation itself was performed correctly."
            )
        return "\n".join(lines)


def verify_result(
    raw: bytes,
    cluster_public: PublicIdentity,
    artifact_directory: Path | None = None,
    expected_job_id: str | None = None,
    expected_circuit_sha256: str | None = None,
) -> VerificationReport:
    """Check a `.aqresult` bundle and report every check individually."""
    checks: list[tuple[str, bool, str]] = []

    try:
        envelope = parse_canonical(raw)
    except CanonicalisationError as exc:
        return VerificationReport(
            ok=False,
            job_id=None,
            cluster_name=None,
            cluster_fingerprint=None,
            checks=[("structure", False, str(exc))],
        )

    if not isinstance(envelope, dict) or envelope.get("schema") != RESULT_ENVELOPE_SCHEMA:
        return VerificationReport(
            ok=False,
            job_id=None,
            cluster_name=None,
            cluster_fingerprint=None,
            checks=[("structure", False, f"unknown schema {envelope.get('schema')!r}")],
        )
    checks.append(("structure", True, "canonical result envelope"))

    payload = envelope.get("manifest")
    signature = envelope.get("signature")
    if not isinstance(payload, dict) or not isinstance(signature, str):
        return VerificationReport(
            ok=False,
            job_id=None,
            cluster_name=envelope.get("cluster_name"),
            cluster_fingerprint=envelope.get("cluster_fingerprint"),
            checks=checks + [("structure", False, "missing manifest or signature")],
        )

    fingerprint_ok = envelope.get("cluster_fingerprint") == cluster_public.signature_fingerprint
    checks.append(
        (
            "signing key",
            fingerprint_ok,
            "bundle names this cluster key"
            if fingerprint_ok
            else f"bundle names {envelope.get('cluster_fingerprint')}, "
            f"verifying against {cluster_public.signature_fingerprint}",
        )
    )

    signature_ok = verify_payload(cluster_public, payload, b64decode(signature))
    checks.append(
        (
            "cluster signature",
            signature_ok,
            "ML-DSA signature verifies" if signature_ok else "signature does not verify",
        )
    )

    try:
        manifest = ResultManifest.from_dict(payload)
    except (ProvenanceError, KeyError, TypeError) as exc:
        checks.append(("manifest", False, f"malformed manifest: {exc}"))
        return VerificationReport(
            ok=False,
            job_id=payload.get("job_id"),
            cluster_name=envelope.get("cluster_name"),
            cluster_fingerprint=envelope.get("cluster_fingerprint"),
            checks=checks,
        )
    checks.append(("manifest", True, f"{len(manifest.artifacts)} artefact(s) described"))

    recomputed_root = compute_output_root(manifest.artifacts)
    root_ok = recomputed_root == manifest.output_merkle_root
    checks.append(
        (
            "merkle root",
            root_ok,
            "root matches the artefact list"
            if root_ok
            else f"recomputed {recomputed_root[:16]}..., signed {manifest.output_merkle_root[:16]}...",
        )
    )

    artifacts_ok = True
    for artifact in manifest.artifacts:
        ok, detail = check_artifact(artifact, artifact_directory)
        artifacts_ok = artifacts_ok and ok
        checks.append((f"artefact {artifact.name}", ok, detail))

    linkage_ok = True
    if expected_job_id is not None:
        ok = manifest.job_id == expected_job_id
        linkage_ok = linkage_ok and ok
        checks.append(
            ("job linkage", ok, "job id matches the submitted bundle" if ok else "job id differs")
        )
    if expected_circuit_sha256 is not None:
        ok = manifest.input_circuit_sha256 == expected_circuit_sha256
        linkage_ok = linkage_ok and ok
        checks.append(
            (
                "input circuit",
                ok,
                "circuit hash matches the submitted job" if ok else "circuit hash differs",
            )
        )

    return VerificationReport(
        ok=all([fingerprint_ok, signature_ok, root_ok, artifacts_ok, linkage_ok]),
        job_id=manifest.job_id,
        cluster_name=envelope.get("cluster_name"),
        cluster_fingerprint=envelope.get("cluster_fingerprint"),
        checks=checks,
        manifest=manifest,
    )


def write_result(path: Path, raw: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def read_result(path: Path) -> bytes:
    path = Path(path)
    if not path.exists():
        raise ProvenanceError(f"result bundle not found: {path}")
    return path.read_bytes()
