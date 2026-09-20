"""Phase 16: replay protection and the verified execution path."""

from __future__ import annotations

import json
import threading

import pytest

from aegisq.algorithms import ghz
from aegisq.cli.main import main
from aegisq.secure.canonical import canonical_bytes
from aegisq.secure.envelope import ExecutionRequest, pack_job, write_job
from aegisq.secure.keys import generate_client_identity, generate_cluster_identity
from aegisq.secure.replay import ReplayDatabase, ReplayDetected
from tests.conftest import requires_liboqs

pytestmark = [pytest.mark.crypto, requires_liboqs()]


@pytest.fixture
def database(tmp_path):
    return ReplayDatabase(tmp_path / "replay.json")


def test_a_fresh_database_is_empty(database):
    assert len(database) == 0
    assert not database.seen("anything")


def test_recording_a_job_makes_it_seen(database):
    database.record("job-1", "nonce", "fp", "2026-01-01T00:00:00Z")
    assert database.seen("job-1")
    assert len(database) == 1


def test_resubmitting_the_same_job_is_refused(database):
    database.record("job-1", "nonce", "fp", "2026-01-01T00:00:00Z")
    with pytest.raises(ReplayDetected, match="already accepted"):
        database.record("job-1", "different-nonce", "fp", "2026-01-01T00:00:00Z")


def test_check_refuses_without_recording(database):
    database.check("job-1")  # not seen yet: no exception
    database.record("job-1", "nonce", "fp", "2026-01-01T00:00:00Z")
    with pytest.raises(ReplayDetected):
        database.check("job-1")


def test_different_jobs_coexist(database):
    for index in range(5):
        database.record(f"job-{index}", "nonce", "fp", "2026-01-01T00:00:00Z")
    assert len(database) == 5


def test_entries_record_who_submitted_and_when(database):
    database.record("job-1", "nonce-value", "client-fp", "2026-01-01T00:00:00Z")
    entry = database.entries()["job-1"]
    assert entry["client_fingerprint"] == "client-fp"
    assert entry["nonce"] == "nonce-value"
    assert entry["created_at"] == "2026-01-01T00:00:00Z"
    assert entry["first_seen"].endswith("Z")


def test_state_survives_a_new_database_object(tmp_path):
    first = ReplayDatabase(tmp_path / "replay.json")
    first.record("job-1", "nonce", "fp", "2026-01-01T00:00:00Z")
    second = ReplayDatabase(tmp_path / "replay.json")
    assert second.seen("job-1")


def test_an_unknown_schema_refuses_to_run(tmp_path):
    path = tmp_path / "replay.json"
    path.write_bytes(canonical_bytes({"schema": "something.else", "entries": {}}))
    with pytest.raises(ReplayDetected, match="unknown schema"):
        ReplayDatabase(path).seen("job-1")


def test_empty_job_id_is_rejected(database):
    with pytest.raises(ValueError, match="job id is required"):
        database.record("", "nonce", "fp", "2026-01-01T00:00:00Z")


def test_concurrent_recording_accepts_exactly_one(tmp_path):
    """Two submissions of one job id must not both succeed."""
    path = tmp_path / "replay.json"
    accepted: list[str] = []
    refused: list[str] = []
    barrier = threading.Barrier(8)

    def submit(index: int) -> None:
        database = ReplayDatabase(path)
        barrier.wait()
        try:
            database.record("contested", f"nonce-{index}", "fp", "2026-01-01T00:00:00Z")
            accepted.append(str(index))
        except ReplayDetected:
            refused.append(str(index))

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(accepted) == 1
    assert len(refused) == 7


def test_pruning_is_explicit_and_reports_what_it_dropped(database):
    database.record("old", "nonce", "fp", "2026-01-01T00:00:00Z")
    assert database.prune(older_than_seconds=0) == 1
    assert len(database) == 0
    # Pruning re-enables replay, which is why it is opt-in.
    database.record("old", "nonce", "fp", "2026-01-01T00:00:00Z")
    assert database.seen("old")


def test_pruning_keeps_recent_entries(database):
    database.record("recent", "nonce", "fp", "2026-01-01T00:00:00Z")
    assert database.prune(older_than_seconds=3600) == 0
    assert database.seen("recent")


def test_environment_variable_overrides_the_path(tmp_path, monkeypatch):
    override = tmp_path / "elsewhere.json"
    monkeypatch.setenv("AEGISQ_REPLAY_DB", str(override))
    database = ReplayDatabase(tmp_path / "ignored.json")
    database.record("job-1", "nonce", "fp", "2026-01-01T00:00:00Z")
    assert override.exists()
    assert not (tmp_path / "ignored.json").exists()


# -- end-to-end through the CLI ---------------------------------------------


@pytest.fixture
def deployment(tmp_path):
    """A client, a cluster, a trusted-key directory and a packed job."""
    client = generate_client_identity("pinar", tmp_path / "client")
    cluster = generate_cluster_identity("courant", tmp_path / "cluster")
    trusted = tmp_path / "trusted"
    trusted.mkdir()
    (trusted / "pinar.public.json").write_bytes(client.public_path.read_bytes())

    bundle = write_job(
        tmp_path / "job.aqjob",
        pack_job(ghz(6), ExecutionRequest(ranks=1, shots=128, seed=3), client, cluster.public),
    )
    return {
        "client": client,
        "cluster": tmp_path / "cluster" / "courant",
        "trusted": trusted,
        "bundle": bundle,
        "replay": tmp_path / "replay.json",
        "result": tmp_path / "result.aqresult",
    }


def run_cli(deployment, extra: list[str] | None = None) -> int:
    return main(
        [
            "secure-run",
            str(deployment["bundle"]),
            "--cluster",
            str(deployment["cluster"]),
            "--trusted",
            str(deployment["trusted"]),
            "--replay-db",
            str(deployment["replay"]),
            *(extra or []),
        ]
    )


def test_secure_run_executes_a_valid_job(deployment, capsys):
    assert run_cli(deployment, ["--output", str(deployment["result"])]) == 0
    out = capsys.readouterr().out
    assert "accepted" in out
    assert "pinar" in out

    # The written file is a signed result bundle, so it is checked the way a
    # recipient would check it rather than by reading raw JSON fields.
    from aegisq.provenance import read_result, verify_result
    from aegisq.secure.canonical import parse_canonical
    from aegisq.secure.keys import load_public_identity

    cluster_public = load_public_identity(deployment["cluster"].with_name("courant.public.json"))
    report = verify_result(read_result(deployment["result"]), cluster_public)
    assert report.ok, report.summary()
    assert report.manifest.execution["world_size"] == 1

    counts_artifact = next(a for a in report.manifest.artifacts if a.name == "counts.json")
    counts = json.loads(counts_artifact.inline.decode())
    assert sum(counts.values()) == 128
    assert set(counts) == {"000000", "111111"}
    assert parse_canonical(counts_artifact.inline) == counts


def test_secure_run_refuses_the_same_bundle_twice(deployment):
    assert run_cli(deployment) == 0
    with pytest.raises(SystemExit, match="already accepted"):
        run_cli(deployment)


def test_secure_run_refuses_an_untrusted_client(tmp_path, deployment):
    empty = tmp_path / "no-trust"
    empty.mkdir()
    (empty / "other.public.json").write_bytes(
        generate_client_identity("other", tmp_path / "other").public_path.read_bytes()
    )
    with pytest.raises(SystemExit, match="not in the cluster's trusted set"):
        main(
            [
                "secure-run",
                str(deployment["bundle"]),
                "--cluster",
                str(deployment["cluster"]),
                "--trusted",
                str(empty),
                "--replay-db",
                str(deployment["replay"]),
            ]
        )


def test_secure_run_refuses_a_tampered_bundle(deployment):
    raw = bytearray(deployment["bundle"].read_bytes())
    index = raw.find(b'"ciphertext":"') + 20
    raw[index] = ord("A") if raw[index] != ord("A") else ord("B")
    deployment["bundle"].write_bytes(bytes(raw))
    with pytest.raises(SystemExit, match="job rejected"):
        run_cli(deployment)


def test_a_rejected_job_is_not_recorded_as_seen(deployment):
    """A refused bundle must not consume its job id."""
    original = deployment["bundle"].read_bytes()
    raw = bytearray(original)
    index = raw.find(b'"signature":"') + 15
    raw[index] = ord("A") if raw[index] != ord("A") else ord("B")
    deployment["bundle"].write_bytes(bytes(raw))
    with pytest.raises(SystemExit):
        run_cli(deployment)

    deployment["bundle"].write_bytes(original)
    assert run_cli(deployment) == 0


def test_secure_run_reports_a_rank_mismatch(tmp_path, deployment, capsys):
    """A job asking for 8 ranks in a 1-rank world runs, and says so."""
    client = deployment["client"]
    from aegisq.secure.keys import load_public_identity

    cluster_public = load_public_identity(deployment["cluster"].with_name("courant.public.json"))
    write_job(
        deployment["bundle"],
        pack_job(ghz(5), ExecutionRequest(ranks=8, shots=16), client, cluster_public),
    )
    assert run_cli(deployment) == 0
    assert "requested 8 rank" in capsys.readouterr().err
