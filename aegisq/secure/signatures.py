"""ML-DSA signatures over canonical bytes.

Two rules the rest of the package relies on:

1. **Signatures are always computed over canonical bytes** produced by
   `aegisq.secure.canonical`, never over an ad-hoc encoding of a dict.
2. **Verification failure is a return value, not an exception path** —
   `verify` returns False for a bad signature and raises only when the inputs
   are structurally unusable. Callers must check the result; `verify_or_raise`
   exists for the common case where a failure should stop everything.
"""

from __future__ import annotations

from typing import Any

from aegisq.secure.canonical import canonical_bytes
from aegisq.secure.keys import (
    IdentityError,
    PublicIdentity,
    SecretIdentity,
    require_oqs,
)


class SignatureError(RuntimeError):
    """Raised when a signature is invalid or cannot be checked."""


def sign_bytes(secret: SecretIdentity, message: bytes) -> bytes:
    """Sign raw bytes with an identity's ML-DSA secret key."""
    oqs = require_oqs()
    try:
        with oqs.Signature(secret.signature_algorithm, secret.signature_secret_key) as signer:
            return signer.sign(message)
    except Exception as exc:
        raise SignatureError(f"signing failed: {exc}") from None


def verify_bytes(public: PublicIdentity, message: bytes, signature: bytes) -> bool:
    """Check a signature. Returns False for a bad signature."""
    oqs = require_oqs()
    if not signature:
        return False
    try:
        with oqs.Signature(public.signature_algorithm) as verifier:
            return bool(verifier.verify(message, signature, public.signature_public_key))
    except Exception:
        # liboqs signals a malformed signature by raising; for the caller that
        # is indistinguishable from "not valid", and must not look like an
        # infrastructure error.
        return False


def sign_payload(secret: SecretIdentity, payload: Any) -> bytes:
    """Sign a JSON-compatible object via its canonical encoding."""
    return sign_bytes(secret, canonical_bytes(payload))


def verify_payload(public: PublicIdentity, payload: Any, signature: bytes) -> bool:
    return verify_bytes(public, canonical_bytes(payload), signature)


def verify_or_raise(public: PublicIdentity, payload: Any, signature: bytes, what: str) -> None:
    """Verify, or stop with a message naming what failed."""
    if not verify_payload(public, payload, signature):
        raise SignatureError(
            f"{what}: signature does not verify against {public.name} "
            f"({public.signature_fingerprint})"
        )


def algorithm_of(identity: PublicIdentity | SecretIdentity) -> str:
    algorithm = identity.signature_algorithm
    if not algorithm:
        raise IdentityError("identity has no signature algorithm")
    return algorithm
