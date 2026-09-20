"""Signed execution provenance: hashing, Merkle trees and result manifests."""

from aegisq.provenance.environment import capture as capture_environment
from aegisq.provenance.hashing import Artifact, check_artifact, describe_bytes, describe_file
from aegisq.provenance.manifest import (
    ProvenanceError,
    ResultManifest,
    VerificationReport,
    artifacts_from_result,
    build_result_manifest,
    read_result,
    sign_result,
    verify_result,
    write_result,
)
from aegisq.provenance.merkle import (
    audit_path,
    leaf_hash,
    merkle_root,
    verify_audit_path,
)

__all__ = [
    "Artifact",
    "describe_bytes",
    "describe_file",
    "check_artifact",
    "ResultManifest",
    "VerificationReport",
    "ProvenanceError",
    "build_result_manifest",
    "artifacts_from_result",
    "sign_result",
    "verify_result",
    "read_result",
    "write_result",
    "capture_environment",
    "merkle_root",
    "leaf_hash",
    "audit_path",
    "verify_audit_path",
]
