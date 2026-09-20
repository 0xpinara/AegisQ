"""Phase 14: post-quantum identity management.

These tests check the *handling* of key material — file permissions,
fingerprints, tamper detection in identity files, domain separation — not the
algorithms, which come from liboqs and are not reimplemented here.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from aegisq.secure import kem, signatures
from aegisq.secure.canonical import (
    CanonicalisationError,
    b64decode,
    b64encode,
    canonical_bytes,
    canonical_hash,
    parse_canonical,
)
from aegisq.secure.keys import (
    KEM_ALGORITHM,
    SIGNATURE_ALGORITHM,
    IdentityError,
    available_algorithms,
    fingerprint,
    generate_client_identity,
    generate_cluster_identity,
    load_identity,
    load_public_identity,
    load_trusted_directory,
)
from tests.conftest import requires_liboqs

pytestmark = [pytest.mark.crypto, requires_liboqs()]


@pytest.fixture
def client(tmp_path):
    return generate_client_identity("pinar", tmp_path / "keys")


@pytest.fixture
def cluster(tmp_path):
    return generate_cluster_identity("courant", tmp_path / "keys")


# -- canonical serialisation ------------------------------------------------


def test_canonical_bytes_sorts_keys_and_drops_whitespace():
    assert canonical_bytes({"b": 1, "a": [1, 2]}) == b'{"a":[1,2],"b":1}'


def test_canonical_form_is_order_independent():
    assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})


def test_non_canonical_input_is_rejected():
    # Same content, different bytes: a verifier must not accept it silently.
    with pytest.raises(CanonicalisationError, match="not in canonical form"):
        parse_canonical(b'{"b":1, "a":2}')


def test_nan_is_rejected():
    with pytest.raises(CanonicalisationError):
        canonical_bytes({"x": float("nan")})


def test_base64_roundtrip_and_validation():
    assert b64decode(b64encode(b"\x00\xff\x10")) == b"\x00\xff\x10"
    with pytest.raises(CanonicalisationError, match="invalid base64"):
        b64decode("not base64!!")


# -- identities -------------------------------------------------------------


def test_required_algorithms_are_available():
    algorithms = available_algorithms()
    assert KEM_ALGORITHM in algorithms["kem"]
    assert SIGNATURE_ALGORITHM in algorithms["signature"]


def test_client_identity_has_a_signing_key_only(client):
    assert client.public.role == "client"
    assert client.public.signature_algorithm == SIGNATURE_ALGORITHM
    assert client.public.kem_public_key is None
    assert client.secret.kem_secret_key is None


def test_cluster_identity_has_both_key_types(cluster):
    assert cluster.public.role == "cluster"
    assert cluster.public.kem_algorithm == KEM_ALGORITHM
    assert cluster.public.kem_public_key is not None
    assert cluster.secret.kem_secret_key is not None


def test_secret_files_are_owner_only(client):
    mode = stat.S_IMODE(os.stat(client.secret_path).st_mode)
    assert mode == 0o600
    assert not mode & (stat.S_IRGRP | stat.S_IROTH)


def test_public_file_contains_no_secret_material(client):
    text = client.public_path.read_text(encoding="utf-8")
    assert "secret" not in text
    assert b64encode(client.secret.signature_secret_key) not in text


def test_fingerprint_is_stable_and_key_dependent(client, cluster):
    assert fingerprint(client.public.signature_public_key) == client.public.signature_fingerprint
    assert client.public.signature_fingerprint != cluster.public.signature_fingerprint
    assert len(client.public.signature_fingerprint) == 32


def test_identity_roundtrips_through_disk(tmp_path, client):
    loaded = load_identity(tmp_path / "keys" / "pinar")
    assert loaded.public.signature_public_key == client.public.signature_public_key
    assert loaded.secret.signature_secret_key == client.secret.signature_secret_key


def test_a_doctored_fingerprint_is_rejected(tmp_path, client):
    payload = json.loads(client.public_path.read_text(encoding="utf-8"))
    payload["signature_fingerprint"] = "0" * 32
    client.public_path.write_bytes(canonical_bytes(payload))
    with pytest.raises(IdentityError, match="fingerprint does not match"):
        load_public_identity(client.public_path)


def test_unknown_schema_is_rejected(tmp_path):
    path = tmp_path / "bogus.public.json"
    path.write_bytes(canonical_bytes({"schema": "other.v1", "name": "x"}))
    with pytest.raises(IdentityError, match="schema"):
        load_public_identity(path)


def test_missing_identity_reports_the_path(tmp_path):
    with pytest.raises(IdentityError, match="not found"):
        load_public_identity(tmp_path / "absent.public.json")


def test_loose_permissions_warn(tmp_path, client):
    os.chmod(client.secret_path, 0o644)
    with pytest.warns(UserWarning, match="readable by group or others"):
        load_identity(tmp_path / "keys" / "pinar")


def test_trusted_directory_indexes_by_fingerprint(tmp_path, client, cluster):
    trusted = load_trusted_directory(tmp_path / "keys")
    assert client.public.signature_fingerprint in trusted
    assert cluster.public.signature_fingerprint in trusted


def test_empty_trusted_directory_is_an_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(IdentityError, match="no public identities"):
        load_trusted_directory(tmp_path / "empty")


# -- KEM, derivation, AEAD --------------------------------------------------


def test_encapsulation_agrees_with_decapsulation(cluster):
    result = kem.encapsulate(cluster.public.kem_public_key)
    recovered = kem.decapsulate(result.ciphertext, cluster.secret.kem_secret_key)
    assert recovered == result.shared_secret
    assert len(result.shared_secret) == 32


def test_decapsulation_with_the_wrong_key_does_not_yield_the_secret(tmp_path, cluster):
    other = generate_cluster_identity("other", tmp_path / "other")
    result = kem.encapsulate(cluster.public.kem_public_key)
    # ML-KEM is designed to return a pseudorandom secret rather than fail, so
    # the check is that the secret differs, not that decapsulation raises.
    wrong = kem.decapsulate(result.ciphertext, other.secret.kem_secret_key)
    assert wrong != result.shared_secret


def test_key_derivation_is_bound_to_the_job(cluster):
    secret = kem.encapsulate(cluster.public.kem_public_key).shared_secret
    assert kem.session_key_for_job(secret, "job-a") != kem.session_key_for_job(secret, "job-b")
    assert kem.session_key_for_job(secret, "job-a") == kem.session_key_for_job(secret, "job-a")
    assert len(kem.session_key_for_job(secret, "job-a")) == 32


def test_empty_job_id_is_rejected(cluster):
    secret = kem.encapsulate(cluster.public.kem_public_key).shared_secret
    with pytest.raises(kem.CryptoError, match="job id"):
        kem.session_key_for_job(secret, "")


def test_aead_roundtrip():
    key = os.urandom(32)
    nonce, ciphertext = kem.encrypt(key, b"payload", b"header")
    assert kem.decrypt(key, nonce, ciphertext, b"header") == b"payload"
    assert len(nonce) == 12


def test_aead_detects_every_kind_of_tampering():
    key = os.urandom(32)
    nonce, ciphertext = kem.encrypt(key, b"payload", b"header")

    flipped = bytearray(ciphertext)
    flipped[0] ^= 0x01
    for label, args in (
        ("ciphertext", (key, nonce, bytes(flipped), b"header")),
        ("associated data", (key, nonce, ciphertext, b"other header")),
        ("key", (os.urandom(32), nonce, ciphertext, b"header")),
        ("nonce", (key, os.urandom(12), ciphertext, b"header")),
    ):
        with pytest.raises(kem.CryptoError, match="authentication failed"):
            kem.decrypt(*args)
        assert label  # each tampering mode is exercised by name


def test_wrong_key_or_nonce_size_is_rejected():
    with pytest.raises(kem.CryptoError, match="32-byte key"):
        kem.encrypt(b"short", b"x", b"")
    with pytest.raises(kem.CryptoError, match="nonce must be"):
        kem.encrypt(os.urandom(32), b"x", b"", nonce=b"short")


def test_nonces_are_unique():
    nonces = {kem.random_nonce() for _ in range(256)}
    assert len(nonces) == 256


# -- signatures -------------------------------------------------------------


def test_signature_verifies_over_canonical_payload(client):
    signature = signatures.sign_payload(client.secret, {"b": 2, "a": 1})
    # Key order must not matter: both encode to the same canonical bytes.
    assert signatures.verify_payload(client.public, {"a": 1, "b": 2}, signature)


def test_modified_payload_fails_verification(client):
    signature = signatures.sign_payload(client.secret, {"a": 1})
    assert not signatures.verify_payload(client.public, {"a": 2}, signature)


def test_modified_signature_fails_verification(client):
    signature = bytearray(signatures.sign_payload(client.secret, {"a": 1}))
    signature[10] ^= 0x01
    assert not signatures.verify_payload(client.public, {"a": 1}, bytes(signature))


def test_truncated_and_empty_signatures_fail(client):
    signature = signatures.sign_payload(client.secret, {"a": 1})
    assert not signatures.verify_payload(client.public, {"a": 1}, signature[:-1])
    assert not signatures.verify_payload(client.public, {"a": 1}, b"")


def test_another_identitys_key_fails_verification(client, cluster):
    signature = signatures.sign_payload(client.secret, {"a": 1})
    assert not signatures.verify_payload(cluster.public, {"a": 1}, signature)


def test_verify_or_raise_names_the_failure(client, cluster):
    signature = signatures.sign_payload(client.secret, {"a": 1})
    with pytest.raises(signatures.SignatureError, match="job envelope"):
        signatures.verify_or_raise(cluster.public, {"a": 1}, signature, "job envelope")


def test_availability_probe_reports_the_suite():
    status = kem.check_available()
    assert status["liboqs"] and status["cryptography"]
    assert status["ml_kem_768"] and status["ml_dsa_65"]
