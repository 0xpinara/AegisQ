"""Merkle trees over result artefacts, following RFC 6962.

Why RFC 6962 rather than "hash pairs until one is left":

* **Domain separation.** Leaves are hashed with a `0x00` prefix and internal
  nodes with `0x01`. Without that, a hash of an internal node can be replayed
  as a leaf, so two different artefact lists can share a root.
* **No duplicated last node.** The tree splits at the largest power of two
  below the leaf count instead of duplicating an odd leaf, which is the
  ambiguity that produced CVE-2012-2459 in Bitcoin — two distinct trees with
  the same root.

The tree lets a verifier confirm that one chunk belongs to a signed result
without re-reading the whole artefact, which matters when the artefact is a
multi-gigabyte state dump.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"

#: Chunk size for splitting large artefacts. 1 MiB keeps the leaf count
#: manageable for multi-gigabyte files while staying small enough that an
#: audit path is cheap to check.
DEFAULT_CHUNK_SIZE = 1 << 20


def leaf_hash(data: bytes) -> bytes:
    """Hash of a leaf, domain-separated from internal nodes."""
    return hashlib.sha256(LEAF_PREFIX + data).digest()


def node_hash(left: bytes, right: bytes) -> bytes:
    """Hash of an internal node."""
    return hashlib.sha256(NODE_PREFIX + left + right).digest()


def _largest_power_of_two_below(n: int) -> int:
    """The split point `k` with `k < n <= 2k`."""
    if n < 2:
        raise ValueError("split point is only defined for n >= 2")
    return 1 << (n - 1).bit_length() - 1


def merkle_root(leaves: Sequence[bytes]) -> bytes:
    """Root over already-hashed leaves (use `leaf_hash` to produce them).

    An empty list hashes to SHA-256 of the empty string, as RFC 6962 defines,
    so "no artefacts" still has a well-defined root.
    """
    if not leaves:
        return hashlib.sha256(b"").digest()
    if len(leaves) == 1:
        return leaves[0]
    split = _largest_power_of_two_below(len(leaves))
    return node_hash(merkle_root(leaves[:split]), merkle_root(leaves[split:]))


def audit_path(leaves: Sequence[bytes], index: int) -> list[bytes]:
    """Sibling hashes proving that `leaves[index]` is in the tree."""
    if not 0 <= index < len(leaves):
        raise IndexError(f"leaf index {index} out of range for {len(leaves)} leaves")
    if len(leaves) == 1:
        return []
    split = _largest_power_of_two_below(len(leaves))
    if index < split:
        return audit_path(leaves[:split], index) + [merkle_root(leaves[split:])]
    return audit_path(leaves[split:], index - split) + [merkle_root(leaves[:split])]


def verify_audit_path(
    leaf: bytes, index: int, tree_size: int, path: Sequence[bytes], root: bytes
) -> bool:
    """Recompute the root from a leaf and its audit path."""
    if not 0 <= index < tree_size:
        return False
    computed = leaf
    node_index = index
    last_index = tree_size - 1
    for sibling in path:
        if node_index % 2 == 1 or node_index == last_index:
            # Right child, or the odd node carried up: sibling is on the left.
            while node_index % 2 == 0 and node_index != 0:
                node_index //= 2
                last_index //= 2
            computed = node_hash(sibling, computed)
        else:
            computed = node_hash(computed, sibling)
        node_index //= 2
        last_index //= 2
    return node_index == 0 and computed == root


def chunks_of(data: bytes, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Iterator[bytes]:
    for offset in range(0, len(data), chunk_size):
        yield data[offset : offset + chunk_size]


def leaves_for_bytes(data: bytes, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[bytes]:
    """Leaf hashes for an in-memory artefact."""
    if not data:
        return [leaf_hash(b"")]
    return [leaf_hash(chunk) for chunk in chunks_of(data, chunk_size)]


def leaves_for_file(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[bytes]:
    """Leaf hashes for a file, streamed so it is never held in memory."""
    leaves: list[bytes] = []
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            leaves.append(leaf_hash(chunk))
    return leaves or [leaf_hash(b"")]


def root_for_bytes(data: bytes, chunk_size: int = DEFAULT_CHUNK_SIZE) -> bytes:
    return merkle_root(leaves_for_bytes(data, chunk_size))


def root_for_file(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> bytes:
    return merkle_root(leaves_for_file(path, chunk_size))


def root_hex(leaves: Iterable[bytes]) -> str:
    return merkle_root(list(leaves)).hex()
