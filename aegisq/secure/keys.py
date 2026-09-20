"""Post-quantum identities for clients and clusters.

Algorithms come from [liboqs](https://openquantumsafe.org/) and are used
exactly as published:

* **ML-KEM-768** (FIPS 203) for key encapsulation,
* **ML-DSA-65** (FIPS 204) for signatures.

Nothing cryptographic is implemented in this file. It handles identity
material: generating key pairs, writing them to disk with sane permissions,
loading them back, and deriving stable fingerprints.

Roles
-----
A **client** submits jobs and needs a signing identity (ML-DSA).
A **cluster** receives jobs and needs both a KEM identity (so clients can
encapsulate to it) and a signing identity (so it can sign results).

Secret and public material live in separate files. Secret files are created
with owner-only permissions and are excluded by `.gitignore`; the loader warns
when it finds one that is group- or world-readable.
"""

from __future__ import annotations

import contextlib
import os
import stat
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aegisq.secure.canonical import b64decode, b64encode, canonical_bytes, sha256_hex

#: Default suite. Both are NIST-standardised (FIPS 203 / FIPS 204).
KEM_ALGORITHM = "ML-KEM-768"
SIGNATURE_ALGORITHM = "ML-DSA-65"

PUBLIC_SCHEMA = "aegisq.identity.public.v1"
SECRET_SCHEMA = "aegisq.identity.secret.v1"


class IdentityError(RuntimeError):
    """Raised when identity material is missing, malformed or unusable."""


def _detach_oqs_stdout_handler(oqs) -> None:
    """Stop liboqs-python from writing to our stdout after import.

    On import `oqs` attaches a `StreamHandler(sys.stdout)` to its own
    logger and sets it to INFO, so every subsequent informational message
    it emits lands on standard output -- which, for this tool, is where
    machine-readable results go.

    The handler is removed rather than silenced so that a caller who has
    configured logging still receives the records by propagation. Only
    handlers writing to stdout are touched; a handler on stderr is a
    diagnostic channel and none of our business.
    """
    import logging
    import sys

    logger = logging.getLogger(oqs.oqs.__name__ if hasattr(oqs, "oqs") else "oqs.oqs")
    for handler in list(logger.handlers):
        writes_to_stdout = getattr(handler, "stream", None) in (sys.stdout, sys.__stdout__)
        if isinstance(handler, logging.StreamHandler) and writes_to_stdout:
            logger.removeHandler(handler)


def require_oqs():
    """Import liboqs, or explain how to get it.

    The import is wrapped in `logging.disable`. liboqs-python logs a line
    about faulthandler while it is being imported, to a handler bound to
    the real `sys.stdout` object -- captured with `from sys import
    stdout`, so `contextlib.redirect_stdout` cannot reach it and the line
    has already been written by the time the import returns.

    The effect was that every `--json` invocation touching the secure
    layer printed

        liboqs-python faulthandler is disabled
        { ... }

    and no JSON parser accepts that. A tool that offers `--json` owns its
    standard output; a dependency's banner on it is a bug in the tool.
    """
    import logging

    previous = logging.root.manager.disable
    try:
        logging.disable(logging.INFO)
        try:
            import oqs  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise IdentityError(
                "liboqs-python is required for the secure job layer; "
                "install it with `pip install aegisq[crypto]`"
            ) from exc
    finally:
        logging.disable(previous)

    _detach_oqs_stdout_handler(oqs)
    return oqs


def available_algorithms() -> dict[str, list[str]]:
    oqs = require_oqs()
    return {
        "kem": sorted(oqs.get_enabled_kem_mechanisms()),
        "signature": sorted(oqs.get_enabled_sig_mechanisms()),
    }


def check_algorithms(kem: str = KEM_ALGORITHM, signature: str = SIGNATURE_ALGORITHM) -> None:
    available = available_algorithms()
    if kem not in available["kem"]:
        raise IdentityError(f"{kem} is not enabled in this liboqs build")
    if signature not in available["signature"]:
        raise IdentityError(f"{signature} is not enabled in this liboqs build")


def fingerprint(public_key: bytes) -> str:
    """Stable short identifier for a public key: first 16 bytes of SHA-256."""
    return sha256_hex(public_key)[:32]


@dataclass(frozen=True)
class PublicIdentity:
    """The shareable half of an identity."""

    name: str
    role: str
    created_at: str
    signature_algorithm: str
    signature_public_key: bytes
    kem_algorithm: str | None = None
    kem_public_key: bytes | None = None

    @property
    def signature_fingerprint(self) -> str:
        return fingerprint(self.signature_public_key)

    @property
    def kem_fingerprint(self) -> str | None:
        return fingerprint(self.kem_public_key) if self.kem_public_key else None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": PUBLIC_SCHEMA,
            "name": self.name,
            "role": self.role,
            "created_at": self.created_at,
            "signature_algorithm": self.signature_algorithm,
            "signature_public_key": b64encode(self.signature_public_key),
            "signature_fingerprint": self.signature_fingerprint,
        }
        if self.kem_public_key is not None:
            payload["kem_algorithm"] = self.kem_algorithm
            payload["kem_public_key"] = b64encode(self.kem_public_key)
            payload["kem_fingerprint"] = self.kem_fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PublicIdentity:
        if payload.get("schema") != PUBLIC_SCHEMA:
            raise IdentityError(f"unknown public identity schema {payload.get('schema')!r}")
        kem_key = payload.get("kem_public_key")
        identity = cls(
            name=str(payload["name"]),
            role=str(payload["role"]),
            created_at=str(payload["created_at"]),
            signature_algorithm=str(payload["signature_algorithm"]),
            signature_public_key=b64decode(str(payload["signature_public_key"])),
            kem_algorithm=payload.get("kem_algorithm"),
            kem_public_key=b64decode(str(kem_key)) if kem_key else None,
        )
        # Fingerprints in the file are a convenience for humans; recompute them
        # so a doctored file cannot advertise a fingerprint it does not have.
        recorded = payload.get("signature_fingerprint")
        if recorded and recorded != identity.signature_fingerprint:
            raise IdentityError("signature fingerprint does not match the public key")
        recorded_kem = payload.get("kem_fingerprint")
        if recorded_kem and recorded_kem != identity.kem_fingerprint:
            raise IdentityError("KEM fingerprint does not match the public key")
        return identity


@dataclass(frozen=True)
class SecretIdentity:
    """The private half. Never transmitted, never committed."""

    name: str
    role: str
    signature_algorithm: str
    signature_secret_key: bytes
    kem_algorithm: str | None = None
    kem_secret_key: bytes | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": SECRET_SCHEMA,
            "name": self.name,
            "role": self.role,
            "signature_algorithm": self.signature_algorithm,
            "signature_secret_key": b64encode(self.signature_secret_key),
        }
        if self.kem_secret_key is not None:
            payload["kem_algorithm"] = self.kem_algorithm
            payload["kem_secret_key"] = b64encode(self.kem_secret_key)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SecretIdentity:
        if payload.get("schema") != SECRET_SCHEMA:
            raise IdentityError(f"unknown secret identity schema {payload.get('schema')!r}")
        kem_key = payload.get("kem_secret_key")
        return cls(
            name=str(payload["name"]),
            role=str(payload["role"]),
            signature_algorithm=str(payload["signature_algorithm"]),
            signature_secret_key=b64decode(str(payload["signature_secret_key"])),
            kem_algorithm=payload.get("kem_algorithm"),
            kem_secret_key=b64decode(str(kem_key)) if kem_key else None,
        )


@dataclass(frozen=True)
class Identity:
    """A public/secret pair together with where it lives on disk."""

    public: PublicIdentity
    secret: SecretIdentity
    public_path: Path | None = None
    secret_path: Path | None = None


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_public(path: Path, identity: PublicIdentity) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(identity.to_dict()))
    return path


def _write_secret(path: Path, secret: SecretIdentity) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create with owner-only permissions from the start, rather than writing
    # the file and tightening it afterwards.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(descriptor, canonical_bytes(secret.to_dict()))
    finally:
        os.close(descriptor)
    # Some filesystems (network shares, Windows) do not honour POSIX modes.
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)
    return path


def generate_client_identity(
    name: str,
    directory: Path,
    signature_algorithm: str = SIGNATURE_ALGORITHM,
) -> Identity:
    """Create a signing identity for a job submitter."""
    oqs = require_oqs()
    check_algorithms(signature=signature_algorithm)

    with oqs.Signature(signature_algorithm) as signer:
        public_key = signer.generate_keypair()
        secret_key = signer.export_secret_key()

    public = PublicIdentity(
        name=name,
        role="client",
        created_at=_now(),
        signature_algorithm=signature_algorithm,
        signature_public_key=public_key,
    )
    secret = SecretIdentity(
        name=name,
        role="client",
        signature_algorithm=signature_algorithm,
        signature_secret_key=secret_key,
    )
    directory = Path(directory)
    return Identity(
        public=public,
        secret=secret,
        public_path=_write_public(directory / f"{name}.public.json", public),
        secret_path=_write_secret(directory / f"{name}.secret.json", secret),
    )


def generate_cluster_identity(
    name: str,
    directory: Path,
    kem_algorithm: str = KEM_ALGORITHM,
    signature_algorithm: str = SIGNATURE_ALGORITHM,
) -> Identity:
    """Create a cluster identity: a KEM key to receive jobs, a signing key to sign results."""
    oqs = require_oqs()
    check_algorithms(kem=kem_algorithm, signature=signature_algorithm)

    with oqs.KeyEncapsulation(kem_algorithm) as kem:
        kem_public = kem.generate_keypair()
        kem_secret = kem.export_secret_key()
    with oqs.Signature(signature_algorithm) as signer:
        sign_public = signer.generate_keypair()
        sign_secret = signer.export_secret_key()

    public = PublicIdentity(
        name=name,
        role="cluster",
        created_at=_now(),
        signature_algorithm=signature_algorithm,
        signature_public_key=sign_public,
        kem_algorithm=kem_algorithm,
        kem_public_key=kem_public,
    )
    secret = SecretIdentity(
        name=name,
        role="cluster",
        signature_algorithm=signature_algorithm,
        signature_secret_key=sign_secret,
        kem_algorithm=kem_algorithm,
        kem_secret_key=kem_secret,
    )
    directory = Path(directory)
    return Identity(
        public=public,
        secret=secret,
        public_path=_write_public(directory / f"{name}.public.json", public),
        secret_path=_write_secret(directory / f"{name}.secret.json", secret),
    )


def load_public_identity(path: Path) -> PublicIdentity:
    from aegisq.secure.canonical import parse_canonical

    path = Path(path)
    if not path.exists():
        raise IdentityError(f"public identity not found: {path}")
    return PublicIdentity.from_dict(parse_canonical(path.read_bytes()))


def load_secret_identity(path: Path) -> SecretIdentity:
    from aegisq.secure.canonical import parse_canonical

    path = Path(path)
    if not path.exists():
        raise IdentityError(f"secret identity not found: {path}")
    _warn_on_loose_permissions(path)
    return SecretIdentity.from_dict(parse_canonical(path.read_bytes()))


def load_identity(base: Path) -> Identity:
    """Load `<base>.public.json` and `<base>.secret.json`."""
    base = Path(base)
    public_path = base.with_name(base.name + ".public.json")
    secret_path = base.with_name(base.name + ".secret.json")
    return Identity(
        public=load_public_identity(public_path),
        secret=load_secret_identity(secret_path),
        public_path=public_path,
        secret_path=secret_path,
    )


def _warn_on_loose_permissions(path: Path) -> None:
    try:
        mode = path.stat().st_mode
    except OSError:  # pragma: no cover
        return
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        warnings.warn(
            f"secret key file {path} is readable by group or others; tighten it with chmod 600",
            stacklevel=3,
        )


def load_trusted_directory(directory: Path) -> dict[str, PublicIdentity]:
    """Load every public identity in a directory, keyed by fingerprint.

    A cluster uses this as its allow-list of client signing keys: a job whose
    signing key is not in the directory is not merely unverified, it is
    unauthorised.
    """
    directory = Path(directory)
    if not directory.exists():
        raise IdentityError(f"trusted key directory not found: {directory}")
    trusted: dict[str, PublicIdentity] = {}
    for path in sorted(directory.glob("*.public.json")):
        identity = load_public_identity(path)
        trusted[identity.signature_fingerprint] = identity
    if not trusted:
        raise IdentityError(f"no public identities found in {directory}")
    return trusted
