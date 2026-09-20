"""Key encapsulation, key derivation and authenticated encryption.

The chain for one job is:

1. **ML-KEM-768** encapsulation against the cluster's public key produces a
   ciphertext and a 32-byte shared secret. Only the holder of the cluster's
   secret key can recover that secret.
2. **HKDF-SHA256** turns the shared secret into an AES key. The KEM output is
   already uniform, but HKDF gives *domain separation*: the `info` string binds
   the derived key to this protocol version and this job id, so a key derived
   for one job can never be used for another.
3. **AES-256-GCM** encrypts the payload with a fresh 96-bit nonce and
   authenticates the envelope's public header as associated data, so the
   header cannot be edited without breaking decryption.

Every primitive comes from liboqs or the `cryptography` package. None is
implemented here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from aegisq.secure.keys import KEM_ALGORITHM, require_oqs

#: AES-256-GCM: 32-byte key, 96-bit nonce (the size GCM is defined for).
AES_KEY_BYTES = 32
AEAD_NONCE_BYTES = 12

#: Bound into the HKDF info string so keys are separated by protocol version.
HKDF_CONTEXT = b"aegisq/pqc/v1"


class CryptoError(RuntimeError):
    """Raised when encapsulation, derivation or decryption fails."""


def require_cryptography():
    try:
        from cryptography.hazmat.primitives import hashes  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise CryptoError(
            "the `cryptography` package is required for the secure job layer; "
            "install it with `pip install aegisq[crypto]`"
        ) from exc


@dataclass(frozen=True)
class Encapsulation:
    """The KEM ciphertext and the secret it carries."""

    ciphertext: bytes
    shared_secret: bytes


def encapsulate(public_key: bytes, algorithm: str = KEM_ALGORITHM) -> Encapsulation:
    """Encapsulate to a recipient's ML-KEM public key."""
    oqs = require_oqs()
    try:
        with oqs.KeyEncapsulation(algorithm) as kem:
            ciphertext, shared_secret = kem.encap_secret(public_key)
    except Exception as exc:  # liboqs raises a variety of low-level errors
        raise CryptoError(f"{algorithm} encapsulation failed: {exc}") from None
    return Encapsulation(ciphertext=ciphertext, shared_secret=shared_secret)


def decapsulate(ciphertext: bytes, secret_key: bytes, algorithm: str = KEM_ALGORITHM) -> bytes:
    """Recover the shared secret with the recipient's ML-KEM secret key."""
    oqs = require_oqs()
    try:
        with oqs.KeyEncapsulation(algorithm, secret_key) as kem:
            return kem.decap_secret(ciphertext)
    except Exception as exc:
        raise CryptoError(f"{algorithm} decapsulation failed: {exc}") from None


def derive_key(
    shared_secret: bytes,
    info: bytes,
    salt: bytes | None = None,
    length: int = AES_KEY_BYTES,
) -> bytes:
    """HKDF-SHA256 from a KEM shared secret to a symmetric key.

    `info` must identify the protocol version and the specific job, so that
    two jobs never derive the same key from the same secret.
    """
    require_cryptography()
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    if not shared_secret:
        raise CryptoError("cannot derive a key from an empty shared secret")
    kdf = HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        info=HKDF_CONTEXT + b"|" + info,
    )
    return kdf.derive(shared_secret)


def random_nonce() -> bytes:
    """A fresh 96-bit AEAD nonce from the OS entropy source."""
    return os.urandom(AEAD_NONCE_BYTES)


def encrypt(key: bytes, plaintext: bytes, associated_data: bytes, nonce: bytes | None = None):
    """AES-256-GCM encryption; returns `(nonce, ciphertext_with_tag)`."""
    require_cryptography()
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if len(key) != AES_KEY_BYTES:
        raise CryptoError(f"AES-256-GCM needs a {AES_KEY_BYTES}-byte key, got {len(key)}")
    nonce = nonce or random_nonce()
    if len(nonce) != AEAD_NONCE_BYTES:
        raise CryptoError(f"AEAD nonce must be {AEAD_NONCE_BYTES} bytes, got {len(nonce)}")
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, associated_data)
    return nonce, ciphertext


def decrypt(key: bytes, nonce: bytes, ciphertext: bytes, associated_data: bytes) -> bytes:
    """AES-256-GCM decryption; raises on any tampering."""
    require_cryptography()
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if len(key) != AES_KEY_BYTES:
        raise CryptoError(f"AES-256-GCM needs a {AES_KEY_BYTES}-byte key, got {len(key)}")
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, associated_data)
    except InvalidTag:
        raise CryptoError(
            "AEAD authentication failed: the ciphertext, nonce, key or protected "
            "header does not match what was encrypted"
        ) from None


def session_key_for_job(shared_secret: bytes, job_id: str) -> bytes:
    """Derive this job's AES key, bound to its identifier."""
    if not job_id:
        raise CryptoError("job id is required for key derivation")
    return derive_key(shared_secret, info=b"job|" + job_id.encode("utf-8"))


def check_available() -> dict[str, bool]:
    """Report whether the secure layer can run here, without raising."""
    status = {"liboqs": False, "cryptography": False, "ml_kem_768": False, "ml_dsa_65": False}
    try:
        from aegisq.secure.keys import available_algorithms

        algorithms = available_algorithms()
        status["liboqs"] = True
        status["ml_kem_768"] = "ML-KEM-768" in algorithms["kem"]
        status["ml_dsa_65"] = "ML-DSA-65" in algorithms["signature"]
    except Exception:  # pragma: no cover - depends on the local liboqs build
        pass
    try:
        require_cryptography()
        status["cryptography"] = True
    except CryptoError:  # pragma: no cover
        pass
    return status
