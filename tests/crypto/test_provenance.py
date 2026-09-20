"""Phase 17: Merkle trees and signed result provenance."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from aegisq.provenance.hashing import check_artifact, describe_bytes, describe_file
from aegisq.provenance.manifest import (
    RESULT_ENVELOPE_SCHEMA,
    ProvenanceError,
    ResultManifest,
    artifacts_from_result,
    build_result_manifest,
    compute_output_root,
    read_result,
    sign_result,
    verify_result,
    write_result,
)
from aegisq.provenance.merkle import (
    audit_path,
    leaf_hash,
    leaves_for_bytes,
    merkle_root,
    node_hash,
    verify_audit_path,
)
from aegisq.secure.canonical import canonical_bytes, parse_canonical
from aegisq.secure.keys import generate_cluster_identity
from tests.conftest import requires_liboqs

pytestmark = pytest.mark.crypto


# -- Merkle tree ------------------------------------------------------------


def test_empty_tree_has_a_defined_root():
    import hashlib

    assert merkle_root([]).hex() == hashlib.sha256(b"").hexdigest()


def test_single_leaf_is_its_own_root():
    leaf = leaf_hash(b"only")
    assert merkle_root([leaf]) == leaf


def test_leaves_and_nodes_are_domain_separated():
    """A node hash must never be reachable as a leaf hash of the same bytes."""
    a, b = leaf_hash(b"a"), leaf_hash(b"b")
    assert node_hash(a, b) != leaf_hash(a + b)
    assert merkle_root([a, b]) == node_hash(a, b)


def test_order_changes_the_root():
    a, b = leaf_hash(b"a"), leaf_hash(b"b")
    assert merkle_root([a, b]) != merkle_root([b, a])


def test_odd_leaf_counts_do_not_collide_with_duplicated_trees():
    """The RFC 6962 split avoids the duplicated-last-leaf ambiguity."""
    leaves = [leaf_hash(bytes([i])) for i in range(3)]
    duplicated = leaves + [leaves[-1]]
    assert merkle_root(leaves) != merkle_root(duplicated)


@pytest.mark.parametrize("size", list(range(1, 33)))
def test_every_leaf_has_a_verifiable_audit_path(size):
    leaves = [leaf_hash(bytes([i])) for i in range(size)]
    root = merkle_root(leaves)
    for index in range(size):
        path = audit_path(leaves, index)
        assert verify_audit_path(leaves[index], index, size, path, root)


@pytest.mark.parametrize("size", [2, 3, 5, 8, 13])
def test_a_wrong_leaf_fails_its_audit_path(size):
    leaves = [leaf_hash(bytes([i])) for i in range(size)]
    root = merkle_root(leaves)
    for index in range(size):
        path = audit_path(leaves, index)
        assert not verify_audit_path(leaf_hash(b"forged"), index, size, path, root)


def test_out_of_range_indices_are_rejected():
    leaves = [leaf_hash(b"a"), leaf_hash(b"b")]
    with pytest.raises(IndexError):
        audit_path(leaves, 5)
    assert not verify_audit_path(leaves[0], 5, 2, [], merkle_root(leaves))


def test_chunking_produces_multiple_leaves():
    data = b"x" * 3000
    leaves = leaves_for_bytes(data, chunk_size=1024)
    assert len(leaves) == 3
    assert merkle_root(leaves) != leaf_hash(data)


# -- artefacts --------------------------------------------------------------


def test_small_artefacts_are_inlined():
    artifact = describe_bytes("counts.json", b'{"00":1}')
    assert artifact.inline == b'{"00":1}'
    assert check_artifact(artifact)[0]


def test_large_artefacts_are_referenced_not_inlined():
    artifact = describe_bytes("big.bin", b"x" * (1 << 17), inline_limit=1 << 16)
    assert artifact.inline is None
    assert artifact.chunk_count >= 1


def test_file_artefacts_are_described_without_loading_them(tmp_path):
    path = tmp_path / "state.bin"
    path.write_bytes(b"y" * 5000)
    artifact = describe_file(path, chunk_size=1024)
    assert artifact.size_bytes == 5000
    assert artifact.chunk_count == 5
    ok, detail = check_artifact(artifact, tmp_path)
    assert ok, detail


def test_a_modified_file_fails_verification(tmp_path):
    path = tmp_path / "state.bin"
    path.write_bytes(b"y" * 1000)
    artifact = describe_file(path)
    path.write_bytes(b"z" * 1000)
    ok, detail = check_artifact(artifact, tmp_path)
    assert not ok
    assert "SHA-256" in detail


def test_a_truncated_file_fails_verification(tmp_path):
    path = tmp_path / "state.bin"
    path.write_bytes(b"y" * 1000)
    artifact = describe_file(path)
    path.write_bytes(b"y" * 999)
    ok, detail = check_artifact(artifact, tmp_path)
    assert not ok
    assert "size" in detail


def test_missing_referenced_file_is_reported_but_not_a_failure(tmp_path):
    artifact = describe_bytes("absent.bin", b"x" * (1 << 17), inline_limit=1)
    ok, detail = check_artifact(artifact, tmp_path)
    assert ok
    assert "not present locally" in detail


def test_a_forged_inline_artefact_fails():
    artifact = describe_bytes("counts.json", b'{"00":1}')
    forged = dataclasses.replace(artifact, inline=b'{"00":999}')
    assert not check_artifact(forged)[0]


def test_missing_file_raises_when_described(tmp_path):
    with pytest.raises(FileNotFoundError):
        describe_file(tmp_path / "nope.bin")


# -- signed result bundles --------------------------------------------------


@pytest.fixture
def cluster(tmp_path):
    return generate_cluster_identity("courant", tmp_path / "cluster")


@pytest.fixture
def manifest():
    return build_result_manifest(
        job_id="job-1",
        input_circuit_sha256="a" * 64,
        artifacts=artifacts_from_result({"00": 512, "11": 512}, {"wall_seconds": 0.12}),
        execution={"world_size": 4, "precision": "fp64"},
        performance={"wall_seconds": 0.12},
        communication={"bytes_sent": 1024},
        client_fingerprint="c" * 32,
        environment={"hostname": "test-host"},
    )


@requires_liboqs()
def test_a_signed_result_verifies(cluster, manifest):
    report = verify_result(sign_result(manifest, cluster), cluster.public)
    assert report.ok
    assert report.job_id == "job-1"
    assert all(ok for _, ok, _ in report.checks)
    assert "not evidence that the computation" in report.summary()


@requires_liboqs()
def test_verification_can_check_job_linkage(cluster, manifest):
    raw = sign_result(manifest, cluster)
    assert verify_result(raw, cluster.public, expected_job_id="job-1").ok
    assert not verify_result(raw, cluster.public, expected_job_id="other").ok
    assert verify_result(raw, cluster.public, expected_circuit_sha256="a" * 64).ok
    assert not verify_result(raw, cluster.public, expected_circuit_sha256="b" * 64).ok


@requires_liboqs()
def test_editing_any_manifest_field_breaks_the_signature(cluster, manifest):
    raw = sign_result(manifest, cluster)
    envelope = parse_canonical(raw)
    envelope["manifest"]["execution"]["world_size"] = 8
    tampered = canonical_bytes(envelope)
    report = verify_result(tampered, cluster.public)
    assert not report.ok
    assert any(name == "cluster signature" and not ok for name, ok, _ in report.checks)


@requires_liboqs()
def test_replacing_an_artefact_breaks_verification(cluster, manifest):
    raw = sign_result(manifest, cluster)
    envelope = parse_canonical(raw)
    envelope["manifest"]["artifacts"][0]["inline"] = "eyJmYWtlIjogMX0="
    report = verify_result(canonical_bytes(envelope), cluster.public)
    assert not report.ok


@requires_liboqs()
def test_reordering_artefacts_changes_the_root(cluster, manifest):
    reordered = dataclasses.replace(manifest, artifacts=list(reversed(manifest.artifacts)))
    assert compute_output_root(reordered.artifacts) != manifest.output_merkle_root


@requires_liboqs()
def test_another_clusters_key_does_not_verify(tmp_path, cluster, manifest):
    other = generate_cluster_identity("other", tmp_path / "other")
    report = verify_result(sign_result(manifest, cluster), other.public)
    assert not report.ok
    assert any(name == "signing key" and not ok for name, ok, _ in report.checks)


@requires_liboqs()
def test_non_canonical_bundles_are_rejected(cluster, manifest):
    import json

    raw = sign_result(manifest, cluster)
    sloppy = json.dumps(json.loads(raw.decode()), indent=1).encode()
    assert not verify_result(sloppy, cluster.public).ok


def test_garbage_input_is_reported_not_crashed(tmp_path):
    from aegisq.secure.keys import generate_cluster_identity as make

    cluster = make("c", tmp_path / "c")
    report = verify_result(b"definitely not a bundle", cluster.public)
    assert not report.ok
    assert report.checks[0][0] == "structure"


@requires_liboqs()
def test_result_bundles_round_trip_through_disk(tmp_path, cluster, manifest):
    path = write_result(tmp_path / "r.aqresult", sign_result(manifest, cluster))
    assert verify_result(read_result(path), cluster.public).ok


def test_missing_result_file_reports_the_path(tmp_path):
    with pytest.raises(ProvenanceError, match="not found"):
        read_result(tmp_path / "absent.aqresult")


@requires_liboqs()
def test_bundle_declares_its_schema(cluster, manifest):
    envelope = parse_canonical(sign_result(manifest, cluster))
    assert envelope["schema"] == RESULT_ENVELOPE_SCHEMA
    assert envelope["manifest"]["schema"] == "aegisq.result.manifest.v1"


def test_unknown_manifest_schema_is_rejected():
    with pytest.raises(ProvenanceError, match="schema"):
        ResultManifest.from_dict({"schema": "other.v1"})


@requires_liboqs()
def test_manifest_records_the_environment(cluster):
    built = build_result_manifest(
        job_id="job-2",
        input_circuit_sha256="b" * 64,
        artifacts=artifacts_from_result({"0": 1}, {}),
        execution={},
        performance={},
        communication={},
    )
    assert built.environment["aegisq_version"]
    assert "git_commit" in built.environment
    assert "mpi_library" in built.environment


@requires_liboqs()
def test_external_artifacts_are_checked_against_disk(tmp_path, cluster):
    payload = b"z" * (1 << 17)
    path: Path = tmp_path / "state.bin"
    path.write_bytes(payload)

    built = build_result_manifest(
        job_id="job-3",
        input_circuit_sha256="c" * 64,
        artifacts=[describe_file(path)],
        execution={},
        performance={},
        communication={},
    )
    raw = sign_result(built, cluster)
    assert verify_result(raw, cluster.public, artifact_directory=tmp_path).ok

    path.write_bytes(b"w" * (1 << 17))
    assert not verify_result(raw, cluster.public, artifact_directory=tmp_path).ok
