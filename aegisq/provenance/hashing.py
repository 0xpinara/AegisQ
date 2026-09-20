"""Artefact descriptors: what was produced, and how to check it later.

An artefact is any file or in-memory blob a job produced. Each one is
described by its length, its SHA-256, and the Merkle root over its chunks.
The plain digest answers "is this byte-for-byte what was signed?"; the Merkle
root additionally allows a single chunk of a large artefact to be checked
without re-reading all of it.

Small artefacts (counts, metrics) are carried **inline** in the result bundle
so it is self-contained. Large ones are referenced by name, and verification
re-reads them from disk when they are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aegisq.provenance.merkle import (
    DEFAULT_CHUNK_SIZE,
    leaves_for_bytes,
    leaves_for_file,
    merkle_root,
)
from aegisq.secure.canonical import b64decode, b64encode, sha256_hex

#: Inline anything at or below this size; reference larger artefacts by name.
INLINE_LIMIT_BYTES = 1 << 16


@dataclass(frozen=True)
class Artifact:
    """A described output of a job."""

    name: str
    media_type: str
    size_bytes: int
    sha256: str
    chunk_size: int
    chunk_count: int
    merkle_root: str
    inline: bytes | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "chunk_size": self.chunk_size,
            "chunk_count": self.chunk_count,
            "merkle_root": self.merkle_root,
        }
        if self.inline is not None:
            payload["inline"] = b64encode(self.inline)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Artifact:
        inline = payload.get("inline")
        return cls(
            name=str(payload["name"]),
            media_type=str(payload["media_type"]),
            size_bytes=int(payload["size_bytes"]),
            sha256=str(payload["sha256"]),
            chunk_size=int(payload["chunk_size"]),
            chunk_count=int(payload["chunk_count"]),
            merkle_root=str(payload["merkle_root"]),
            inline=b64decode(str(inline)) if inline is not None else None,
        )


def describe_bytes(
    name: str,
    data: bytes,
    media_type: str = "application/json",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    inline_limit: int = INLINE_LIMIT_BYTES,
) -> Artifact:
    """Describe an in-memory artefact, inlining it when it is small."""
    leaves = leaves_for_bytes(data, chunk_size)
    return Artifact(
        name=name,
        media_type=media_type,
        size_bytes=len(data),
        sha256=sha256_hex(data),
        chunk_size=chunk_size,
        chunk_count=len(leaves),
        merkle_root=merkle_root(leaves).hex(),
        inline=data if len(data) <= inline_limit else None,
    )


def describe_file(
    path: Path,
    media_type: str = "application/octet-stream",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    name: str | None = None,
) -> Artifact:
    """Describe a file on disk without loading it into memory."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"artefact not found: {path}")
    leaves = leaves_for_file(path, chunk_size)
    from aegisq.secure.canonical import sha256_file

    return Artifact(
        name=name or path.name,
        media_type=media_type,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        chunk_size=chunk_size,
        chunk_count=len(leaves),
        merkle_root=merkle_root(leaves).hex(),
        inline=None,
    )


def check_artifact(artifact: Artifact, directory: Path | None = None) -> tuple[bool, str]:
    """Re-derive an artefact's hashes and compare them with its description.

    Returns `(ok, explanation)` rather than raising, so a verifier can report
    every artefact rather than stopping at the first problem.
    """
    if artifact.inline is not None:
        data = artifact.inline
        if len(data) != artifact.size_bytes:
            return False, f"{artifact.name}: inline length {len(data)} != {artifact.size_bytes}"
        if sha256_hex(data) != artifact.sha256:
            return False, f"{artifact.name}: inline content does not match its SHA-256"
        leaves = leaves_for_bytes(data, artifact.chunk_size)
        if merkle_root(leaves).hex() != artifact.merkle_root:
            return False, f"{artifact.name}: inline content does not match its Merkle root"
        return True, f"{artifact.name}: inline content verified"

    if directory is None:
        return True, f"{artifact.name}: referenced only (no artefact directory given)"
    path = Path(directory) / artifact.name
    if not path.exists():
        return True, f"{artifact.name}: not present locally, hashes not checked"
    from aegisq.secure.canonical import sha256_file

    if path.stat().st_size != artifact.size_bytes:
        return False, f"{artifact.name}: size on disk differs from the signed description"
    if sha256_file(path) != artifact.sha256:
        return False, f"{artifact.name}: content does not match its signed SHA-256"
    if merkle_root(leaves_for_file(path, artifact.chunk_size)).hex() != artifact.merkle_root:
        return False, f"{artifact.name}: content does not match its signed Merkle root"
    return True, f"{artifact.name}: file verified against the signed description"
