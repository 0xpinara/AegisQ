"""Canonical byte serialisation for anything that gets signed or hashed.

A signature is only meaningful over an unambiguous byte string. If the same
logical object can serialise two ways, a verifier and a signer can disagree
about what was signed — which is how signature schemes get bypassed without
anyone breaking the cryptography.

Rules, applied everywhere in AegisQ:

* UTF-8 JSON with sorted keys and no insignificant whitespace,
* no NaN or Infinity (they are not valid JSON and round-trip badly),
* binary values carried as base64 strings, never as raw bytes,
* the serialised form is what is signed, hashed and transmitted; a verifier
  re-serialises the parsed object and checks it matches the bytes it received.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from typing import Any


class CanonicalisationError(ValueError):
    """Raised when a value cannot be represented canonically."""


def canonical_bytes(payload: Any) -> bytes:
    """Deterministic UTF-8 encoding of a JSON-compatible object."""
    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalisationError(f"value is not canonically serialisable: {exc}") from None
    return text.encode("utf-8")


def canonical_hash(payload: Any) -> str:
    """SHA-256 of the canonical encoding, as lowercase hex."""
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def parse_canonical(raw: bytes) -> Any:
    """Parse canonical bytes and verify they *were* canonical.

    Re-serialising and comparing rejects a payload whose bytes differ from the
    canonical form of its own content — for example one with reordered keys or
    injected whitespace, which could otherwise carry a valid signature over
    different bytes than a naive verifier would reconstruct.
    """
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalisationError(f"payload is not valid UTF-8 JSON: {exc}") from None
    if canonical_bytes(payload) != raw:
        raise CanonicalisationError("payload is not in canonical form")
    return payload


def b64encode(data: bytes) -> str:
    """Base64 without newlines, for embedding binary in JSON."""
    return base64.b64encode(data).decode("ascii")


def b64decode(text: str) -> bytes:
    try:
        return base64.b64decode(text.encode("ascii"), validate=True)
    except (binascii.Error, UnicodeEncodeError, ValueError) as exc:
        raise CanonicalisationError(f"invalid base64 field: {exc}") from None


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed so large artefacts do not sit in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
