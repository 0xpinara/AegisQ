"""Phase 15: job envelopes, and every way of tampering with one.

The security claim is narrow and testable: a bundle that has been modified in
*any* way, or that was not signed by a trusted client, or that was addressed
to a different cluster, must fail to open. Each of those is a separate test
here, because "it works on the happy path" is not a security property.
"""

from __future__ import annotations

import json

import pytest

from aegisq.algorithms import ghz, qft
from aegisq.circuit import Circuit
from aegisq.secure import kem
from aegisq.secure.canonical import b64decode, b64encode, canonical_bytes, parse_canonical
from aegisq.secure.envelope import (
    CRYPTO_SUITE,
    EnvelopeError,
    ExecutionRequest,
    circuit_digest,
    inspect_job,
    open_job,
    pack_job,
    read_job,
    write_job,
)
from aegisq.secure.keys import generate_client_identity, generate_cluster_identity
from aegisq.secure.signatures import SignatureError
from tests.conftest import requires_liboqs

pytestmark = [pytest.mark.crypto, requires_liboqs()]


@pytest.fixture
def parties(tmp_path):
    client = generate_client_identity("pinar", tmp_path / "client")
    cluster = generate_cluster_identity("courant", tmp_path / "cluster")
    trusted = {client.public.signature_fingerprint: client.public}
    return client, cluster, trusted


@pytest.fixture
def bundle(parties):
    client, cluster, _ = parties
    return pack_job(ghz(8), ExecutionRequest(ranks=4, shots=256, seed=7), client, cluster.public)


def mutate(raw: bytes, path: list[str], value) -> bytes:
    """Rewrite one field of an envelope, keeping it canonical."""
    envelope = parse_canonical(raw)
    node = envelope
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    return canonical_bytes(envelope)


# -- happy path -------------------------------------------------------------


def test_a_valid_bundle_opens(parties, bundle):
    _, cluster, trusted = parties
    opened = open_job(bundle, cluster, trusted)
    assert opened.circuit.name == "ghz8"
    assert opened.manifest.execution.ranks == 4
    assert opened.manifest.execution.shots == 256
    assert opened.client.name == "pinar"


def test_circuit_survives_the_round_trip_exactly(parties):
    client, cluster, trusted = parties
    original = qft(6)
    raw = pack_job(original, ExecutionRequest(), client, cluster.public)
    restored = open_job(raw, cluster, trusted).circuit
    assert restored.to_dict() == original.to_dict()
    assert circuit_digest(restored) == circuit_digest(original)


def test_manifest_records_the_circuit_hash(parties, bundle):
    _, cluster, trusted = parties
    opened = open_job(bundle, cluster, trusted)
    assert opened.manifest.circuit_sha256 == circuit_digest(opened.circuit)
    assert opened.manifest.num_qubits == 8
    assert opened.manifest.nonce  # replay token present


def test_each_bundle_has_a_unique_job_id_and_nonce(parties):
    client, cluster, trusted = parties
    first = open_job(pack_job(ghz(4), ExecutionRequest(), client, cluster.public), cluster, trusted)
    second = open_job(
        pack_job(ghz(4), ExecutionRequest(), client, cluster.public), cluster, trusted
    )
    assert first.manifest.job_id != second.manifest.job_id
    assert first.manifest.nonce != second.manifest.nonce


def test_bundles_are_written_and_read_from_disk(tmp_path, parties, bundle):
    _, cluster, trusted = parties
    path = write_job(tmp_path / "job.aqjob", bundle)
    assert open_job(read_job(path), cluster, trusted).circuit.num_qubits == 8


def test_missing_bundle_reports_the_path(tmp_path):
    with pytest.raises(EnvelopeError, match="not found"):
        read_job(tmp_path / "absent.aqjob")


# -- confidentiality --------------------------------------------------------


def test_the_circuit_is_not_visible_in_the_bundle(parties):
    client, cluster, _ = parties
    circuit = Circuit(4, name="secret-circuit").h(0).cx(0, 1).rz(2, 0.1234567)
    raw = pack_job(circuit, ExecutionRequest(), client, cluster.public)
    assert b"secret-circuit" not in raw
    assert b"0.1234567" not in raw

    # Nothing about the program appears outside the ciphertext: the header
    # carries only routing and crypto metadata.
    header = parse_canonical(raw)["protected"]["header"]
    assert set(header) == {
        "schema",
        "job_id",
        "created_at",
        "crypto_suite",
        "client_fingerprint",
        "client_name",
        "cluster_fingerprint",
        "cluster_name",
        "kem_ciphertext",
        "aead_nonce",
        "payload_bytes",
    }


def test_inspect_exposes_metadata_but_not_the_payload(bundle):
    summary = inspect_job(bundle)
    assert summary["client_name"] == "pinar"
    assert summary["cluster_name"] == "courant"
    assert summary["crypto_suite"] == CRYPTO_SUITE
    assert summary["verified"] is False
    assert "circuit" not in summary
    assert "not been checked" in summary["note"]


def test_inspect_does_not_require_any_key(bundle):
    # Deliberately no identity arguments: an operator can triage a file
    # before deciding whether to run it.
    assert inspect_job(bundle)["job_id"]


# -- tampering --------------------------------------------------------------


def test_flipping_a_ciphertext_byte_is_rejected(parties, bundle):
    _, cluster, trusted = parties
    envelope = parse_canonical(bundle)
    ciphertext = bytearray(b64decode(envelope["protected"]["ciphertext"]))
    ciphertext[5] ^= 0x01
    tampered = mutate(bundle, ["protected", "ciphertext"], b64encode(bytes(ciphertext)))
    with pytest.raises(SignatureError, match="does not verify"):
        open_job(tampered, cluster, trusted)


def test_flipping_a_signature_byte_is_rejected(parties, bundle):
    _, cluster, trusted = parties
    envelope = parse_canonical(bundle)
    signature = bytearray(b64decode(envelope["signature"]))
    signature[20] ^= 0x01
    tampered = mutate(bundle, ["signature"], b64encode(bytes(signature)))
    with pytest.raises(SignatureError, match="does not verify"):
        open_job(tampered, cluster, trusted)


@pytest.mark.parametrize(
    "field, value",
    [
        ("job_id", "0" * 32),
        ("created_at", "1999-01-01T00:00:00Z"),
        ("client_name", "someone-else"),
        ("payload_bytes", 1),
    ],
)
def test_editing_a_protected_header_field_is_rejected(parties, bundle, field, value):
    _, cluster, trusted = parties
    tampered = mutate(bundle, ["protected", "header", field], value)
    with pytest.raises((SignatureError, EnvelopeError)):
        open_job(tampered, cluster, trusted)


def test_stripping_the_signature_is_rejected(parties, bundle):
    _, cluster, trusted = parties
    tampered = mutate(bundle, ["signature"], b64encode(b""))
    with pytest.raises(SignatureError):
        open_job(tampered, cluster, trusted)


def test_a_non_canonical_envelope_is_rejected(parties, bundle):
    _, cluster, trusted = parties
    # Same content, whitespace injected: must not be accepted.
    sloppy = json.dumps(json.loads(bundle.decode()), indent=2).encode()
    with pytest.raises(EnvelopeError, match="canonical"):
        open_job(sloppy, cluster, trusted)


def test_garbage_input_is_rejected(parties):
    _, cluster, trusted = parties
    with pytest.raises(EnvelopeError):
        open_job(b"not an envelope at all", cluster, trusted)


def test_unknown_envelope_schema_is_rejected(parties, bundle):
    _, cluster, trusted = parties
    with pytest.raises(EnvelopeError, match="schema"):
        open_job(mutate(bundle, ["schema"], "aegisq.job.envelope.v99"), cluster, trusted)


def test_downgraded_crypto_suite_is_rejected(parties, bundle):
    _, cluster, trusted = parties
    weak = dict(CRYPTO_SUITE, kem="RSA-2048")
    tampered = mutate(bundle, ["protected", "header", "crypto_suite"], weak)
    with pytest.raises((SignatureError, EnvelopeError)):
        open_job(tampered, cluster, trusted)


# -- authorisation ----------------------------------------------------------


def test_an_untrusted_client_is_rejected_before_any_decryption(tmp_path, parties):
    _, cluster, _ = parties
    stranger = generate_client_identity("stranger", tmp_path / "stranger")
    raw = pack_job(ghz(4), ExecutionRequest(), stranger, cluster.public)
    with pytest.raises(EnvelopeError, match="not in the cluster's trusted set"):
        open_job(raw, cluster, {})


def test_a_bundle_for_another_cluster_is_rejected(tmp_path, parties):
    client, _, trusted = parties
    other_cluster = generate_cluster_identity("other", tmp_path / "other")
    raw = pack_job(ghz(4), ExecutionRequest(), client, other_cluster.public)
    with pytest.raises(EnvelopeError, match="different cluster key"):
        open_job(raw, _cluster_from(tmp_path, "third"), trusted)


def _cluster_from(tmp_path, name):
    return generate_cluster_identity(name, tmp_path / name)


def test_the_wrong_cluster_secret_cannot_decrypt(tmp_path, parties):
    client, cluster, trusted = parties
    raw = pack_job(ghz(4), ExecutionRequest(), client, cluster.public)

    # Same advertised fingerprint, different secret key: decryption must fail
    # rather than produce a plausible-looking payload.
    impostor = generate_cluster_identity("impostor", tmp_path / "impostor")
    forged = mutate(
        raw, ["protected", "header", "cluster_fingerprint"], impostor.public.kem_fingerprint
    )
    with pytest.raises((SignatureError, EnvelopeError, kem.CryptoError)):
        open_job(forged, impostor, trusted)


def test_packing_to_a_client_identity_is_refused(parties):
    client, _, _ = parties
    with pytest.raises(EnvelopeError, match="no KEM key"):
        pack_job(ghz(4), ExecutionRequest(), client, client.public)


def test_packing_with_a_cluster_identity_as_the_client_is_refused(parties):
    _, cluster, _ = parties
    with pytest.raises(EnvelopeError, match="not a client identity"):
        pack_job(ghz(4), ExecutionRequest(), cluster, cluster.public)
