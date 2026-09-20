"""Replay protection for submitted job bundles.

A signed bundle stays valid forever — that is what a signature means. Without
additional state, anyone who captures one can resubmit it and the cluster will
happily run it again, burning allocation and producing a second result that
looks just as authentic as the first.

The defence is a record of job identifiers the cluster has already accepted.
It is deliberately small: a JSON file holding one entry per job, guarded by an
exclusive lock for the read-modify-write.

Honest limitations
------------------
* The state file is per-cluster-installation. Two clusters sharing a client do
  not share replay state, and neither do two installations pointed at
  different files.
* Locking uses `fcntl.flock`, which is reliable on a local POSIX filesystem.
  On NFS its behaviour depends on the mount and the server; a deployment that
  needs multi-node submission safety should point `AEGISQ_REPLAY_DB` at local
  storage on the submission host or use a real database.
* Pruning old entries is opt-in. An unbounded file is the safe default,
  because forgetting a job id is exactly what an attacker wants.
"""

from __future__ import annotations

import calendar
import contextlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aegisq.secure.canonical import canonical_bytes, parse_canonical

REPLAY_SCHEMA = "aegisq.replay.v1"
DEFAULT_PATH = Path("replay_db.json")


class ReplayDetected(RuntimeError):
    """Raised when a job identifier has already been accepted."""


@dataclass(frozen=True)
class ReplayEntry:
    job_id: str
    nonce: str
    client_fingerprint: str
    created_at: str
    first_seen: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "nonce": self.nonce,
            "client_fingerprint": self.client_fingerprint,
            "created_at": self.created_at,
            "first_seen": self.first_seen,
        }


class ReplayDatabase:
    """A file-backed set of accepted job identifiers."""

    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self.path = Path(os.environ.get("AEGISQ_REPLAY_DB", path))

    # -- storage ------------------------------------------------------------

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema": REPLAY_SCHEMA, "entries": {}}
        payload = parse_canonical(self.path.read_bytes())
        if payload.get("schema") != REPLAY_SCHEMA:
            raise ReplayDetected(
                f"replay database {self.path} has an unknown schema "
                f"{payload.get('schema')!r}; refusing to run rather than "
                "accepting jobs without replay protection"
            )
        return payload

    def _write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_bytes(canonical_bytes(payload))
        # Atomic replace so a crash mid-write cannot leave a truncated file
        # that would silently lose replay state.
        os.replace(temporary, self.path)

    @contextlib.contextmanager
    def _locked(self):
        """Hold an exclusive lock for the whole read-modify-write."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with open(lock_path, "a+") as handle:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):  # pragma: no cover - platform dependent
                # Without flock the check-and-set is still correct for a single
                # writer; only concurrent submission becomes racy.
                pass
            yield

    # -- queries ------------------------------------------------------------

    def seen(self, job_id: str) -> bool:
        return job_id in self._read()["entries"]

    def entries(self) -> dict[str, dict[str, Any]]:
        return dict(self._read()["entries"])

    def __len__(self) -> int:
        return len(self._read()["entries"])

    # -- mutation -----------------------------------------------------------

    def record(
        self,
        job_id: str,
        nonce: str,
        client_fingerprint: str,
        created_at: str,
    ) -> ReplayEntry:
        """Accept a job id, or refuse it because it has been seen before."""
        if not job_id:
            raise ValueError("job id is required")
        with self._locked():
            payload = self._read()
            existing = payload["entries"].get(job_id)
            if existing is not None:
                raise ReplayDetected(
                    f"job {job_id} was already accepted at {existing['first_seen']} "
                    f"from client {existing['client_fingerprint']}; "
                    "resubmitting a signed bundle is not permitted"
                )
            entry = ReplayEntry(
                job_id=job_id,
                nonce=nonce,
                client_fingerprint=client_fingerprint,
                created_at=created_at,
                first_seen=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            )
            payload["entries"][job_id] = entry.to_dict()
            self._write(payload)
            return entry

    def check(self, job_id: str) -> None:
        """Raise if a job id has already been accepted, without recording it.

        Used by non-writing MPI ranks: every rank refuses a replayed job, but
        only rank 0 mutates the state file.
        """
        entries = self._read()["entries"]
        if job_id in entries:
            raise ReplayDetected(
                f"job {job_id} was already accepted at {entries[job_id]['first_seen']}"
            )

    def prune(self, older_than_seconds: float) -> int:
        """Drop entries older than a cutoff. Opt-in, and unsafe by nature.

        Forgetting a job id re-enables replaying it, so this exists only for
        operators who have an external reason to bound the file, and returns
        the number of entries dropped so the decision is visible.
        """
        cutoff = time.time() - older_than_seconds
        removed = 0
        with self._locked():
            payload = self._read()
            keep = {}
            for job_id, entry in payload["entries"].items():
                try:
                    # timegm, not mktime: first_seen is UTC, and mktime would
                    # reinterpret it in local time and shift the cutoff.
                    stamp = calendar.timegm(
                        time.strptime(entry["first_seen"], "%Y-%m-%dT%H:%M:%SZ")
                    )
                except (KeyError, ValueError):
                    keep[job_id] = entry
                    continue
                if stamp >= cutoff:
                    keep[job_id] = entry
                else:
                    removed += 1
            payload["entries"] = keep
            self._write(payload)
        return removed
