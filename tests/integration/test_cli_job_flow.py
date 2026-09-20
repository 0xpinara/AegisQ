"""The secure job flow, driven through the command line.

Every other test in this repository imports the library. These do not:
they invoke `aegisq` as a subprocess, the way a user does, and follow one
job from key generation through packing, inspection, execution and result
verification.

That distinction matters for the security claims. The library can be
correct while the CLI hands it the wrong file, checks in the wrong order,
or exits zero on a rejection -- and a rejection that exits zero is
indistinguishable from an acceptance to any script that wraps this tool.
So the assertions here are about observable behaviour: exit status, what
reaches stdout, and what a party without the secret keys can learn.

Cheap negative cases only. This is not a cryptographic evaluation; it
checks that the CLI actually refuses what the threat model says it
refuses, and refuses it loudly.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import stat
import subprocess
import sys

import pytest

from tests.conftest import requires_liboqs

pytestmark = requires_liboqs()


def aegisq(*args, cwd=None):
    """Invoke the CLI the way a user would, and keep the exit status."""
    return subprocess.run(
        [sys.executable, "-m", "aegisq.cli.main", *map(str, args)],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )


def ok(completed, what: str):
    assert completed.returncode == 0, f"{what} failed:\n{completed.stdout}\n{completed.stderr}"
    return completed


def parse_json(completed):
    """The CLI's JSON output, ignoring what liboqs prints to stderr."""
    return json.loads(completed.stdout)


@pytest.fixture(scope="module")
def identities(tmp_path_factory):
    """One client and one cluster, generated once for the whole module."""
    directory = tmp_path_factory.mktemp("identities")
    ok(aegisq("keys", "init-client", "alice", "--directory", directory), "init-client")
    ok(aegisq("keys", "init-cluster", "hpc1", "--directory", directory), "init-cluster")

    trusted = directory / "trusted"
    trusted.mkdir()
    shutil.copy(directory / "alice.public.json", trusted / "alice.public.json")
    return {
        "dir": directory,
        "client": directory / "alice",
        "cluster": directory / "hpc1",
        "cluster_public": directory / "hpc1.public.json",
        "trusted": trusted,
    }


@pytest.fixture
def bundle(identities, tmp_path):
    """A freshly packed job, so replay state never leaks between tests."""
    path = tmp_path / "job.aqj"
    ok(
        aegisq(
            "secure-pack",
            "ghz",
            "--qubits",
            4,
            "--shots",
            128,
            "--seed",
            7,
            "--identity",
            identities["client"],
            "--cluster",
            identities["cluster_public"],
            "--output",
            path,
        ),
        "secure-pack",
    )
    return path


def run_bundle(identities, bundle, tmp_path, name="run"):
    return aegisq(
        "secure-run",
        bundle,
        "--cluster",
        identities["cluster"],
        "--trusted",
        identities["trusted"],
        "--replay-db",
        tmp_path / f"{name}.db",
        "--output",
        tmp_path / f"{name}.result.json",
        "--json",
    )


def test_a_job_keeps_its_identity_from_packing_to_verification(identities, bundle, tmp_path):
    """The same job id and circuit hash must appear at every stage.

    This is the end-to-end property the individual unit tests cannot
    state: packing, inspection, execution and result verification each
    handle the job separately, and nothing catches a mix-up between them
    unless the identifiers are compared across all four.
    """
    inspected = parse_json(ok(aegisq("secure-inspect", bundle, "--json"), "secure-inspect"))
    executed = parse_json(ok(run_bundle(identities, bundle, tmp_path), "secure-run"))

    assert inspected["job_id"] == executed["job_id"]
    assert inspected["client_fingerprint"] == executed["client"]["fingerprint"]

    verified = parse_json(
        ok(
            aegisq(
                "verify-result",
                tmp_path / "run.result.json",
                "--cluster",
                identities["cluster_public"],
                "--job-id",
                executed["job_id"],
                "--circuit-sha256",
                executed["circuit"]["sha256"],
                "--json",
            ),
            "verify-result",
        )
    )
    assert all(check["ok"] for check in verified["checks"]), verified


def test_a_ghz_state_survives_the_whole_pipeline(identities, bundle, tmp_path):
    """The point of the envelope is to deliver a computation intact.

    A 4-qubit GHZ state may only be measured as all-zeros or all-ones, so
    any corruption of the circuit between packing and execution shows up
    as a third key in the counts.
    """
    executed = parse_json(ok(run_bundle(identities, bundle, tmp_path), "secure-run"))
    assert set(executed["counts"]) <= {"0000", "1111"}
    assert sum(executed["counts"].values()) == 128


def test_inspection_reveals_metadata_but_not_the_payload(bundle):
    """A bundle in transit must not disclose what it computes.

    `secure-inspect` is what an operator can run without any key at all,
    so it is the right place to check what the envelope leaks. It must
    also not imply more than it knows: the signature has not been checked
    at this point.
    """
    inspected = parse_json(ok(aegisq("secure-inspect", bundle, "--json"), "secure-inspect"))

    assert inspected["verified"] is False
    assert "ghz" not in json.dumps(inspected).lower()
    for leaked in ("circuit", "qubits", "shots", "seed"):
        assert leaked not in inspected, f"inspection disclosed {leaked}"


def test_a_replayed_bundle_is_rejected_with_a_failing_exit_code(identities, bundle, tmp_path):
    """Replay protection that exits zero protects nothing a script can see."""
    ok(run_bundle(identities, bundle, tmp_path, name="first"), "first execution")

    again = aegisq(
        "secure-run",
        bundle,
        "--cluster",
        identities["cluster"],
        "--trusted",
        identities["trusted"],
        "--replay-db",
        tmp_path / "first.db",
    )
    assert again.returncode != 0, "a replayed job exited zero"
    assert "already accepted" in (again.stdout + again.stderr)


def test_a_tampered_ciphertext_is_rejected(identities, bundle, tmp_path):
    """Flip one bit of the payload, not of the file's framing.

    Appending junk to the bundle only proves the JSON parser works. This
    edits a byte inside the AES-GCM ciphertext and leaves the envelope
    perfectly well-formed, so the rejection has to come from the
    signature or the AEAD tag.
    """
    envelope = json.loads(bundle.read_text())
    raw = bytearray(base64.b64decode(envelope["protected"]["ciphertext"]))
    raw[0] ^= 0x01
    envelope["protected"]["ciphertext"] = base64.b64encode(bytes(raw)).decode()
    bundle.write_text(json.dumps(envelope))

    rejected = run_bundle(identities, bundle, tmp_path, name="tampered")
    assert rejected.returncode != 0, "a tampered bundle was executed"


def test_an_untrusted_client_is_rejected(identities, bundle, tmp_path):
    """A valid signature from an unknown key is still an unknown key."""
    empty = tmp_path / "nobody-trusted"
    empty.mkdir()

    rejected = aegisq(
        "secure-run",
        bundle,
        "--cluster",
        identities["cluster"],
        "--trusted",
        empty,
        "--replay-db",
        tmp_path / "untrusted.db",
    )
    assert rejected.returncode != 0, "a job from an untrusted client was executed"


def test_a_bundle_addressed_to_another_cluster_is_rejected(identities, bundle, tmp_path):
    """Being able to decrypt is not the same as being the addressee."""
    other = tmp_path / "other-cluster"
    other.mkdir()
    ok(aegisq("keys", "init-cluster", "hpc2", "--directory", other), "second cluster")

    rejected = aegisq(
        "secure-run",
        bundle,
        "--cluster",
        other / "hpc2",
        "--trusted",
        identities["trusted"],
        "--replay-db",
        tmp_path / "misaddressed.db",
    )
    assert rejected.returncode != 0, "a misaddressed bundle was executed"


def test_result_verification_detects_an_edited_count(identities, bundle, tmp_path):
    """The signed record must not survive having its numbers changed.

    Note what this does and does not establish: it shows the record was
    not altered after the cluster signed it. It says nothing about whether
    the cluster computed the right answer in the first place.
    """
    ok(run_bundle(identities, bundle, tmp_path, name="edit"), "secure-run")
    result_path = tmp_path / "edit.result.json"

    record = json.loads(result_path.read_text())
    counts_artifact = next(
        artifact
        for artifact in record["manifest"]["artifacts"]
        if artifact["name"] == "counts.json"
    )
    counts = json.loads(base64.b64decode(counts_artifact["inline"]))
    # Move one shot from one outcome to the other: a result that is still
    # a plausible GHZ measurement, and differs only in the numbers.
    outcomes = sorted(counts)
    counts[outcomes[0]] += 1
    counts[outcomes[-1]] -= 1
    counts_artifact["inline"] = base64.b64encode(
        json.dumps(counts, separators=(",", ":")).encode()
    ).decode()
    result_path.write_text(json.dumps(record))

    checked = aegisq(
        "verify-result", result_path, "--cluster", identities["cluster_public"], "--json"
    )
    assert checked.returncode != 0, "an edited result verified successfully"


def test_job_id_binding_is_enforced_by_verify_result(identities, bundle, tmp_path):
    """Verifying against the wrong job must fail, or the binding is decorative."""
    ok(run_bundle(identities, bundle, tmp_path, name="binding"), "secure-run")

    checked = aegisq(
        "verify-result",
        tmp_path / "binding.result.json",
        "--cluster",
        identities["cluster_public"],
        "--job-id",
        "0" * 32,
    )
    assert checked.returncode != 0, "a result verified against the wrong job id"


def test_secret_key_files_are_not_readable_by_others(identities):
    """The CLI promises mode 0600 in its own help text."""
    for name in ("alice.secret.json", "hpc1.secret.json"):
        mode = stat.S_IMODE(os.stat(identities["dir"] / name).st_mode)
        assert not mode & (stat.S_IRWXG | stat.S_IRWXO), f"{name} is mode {mode:04o}"


def test_the_same_seed_gives_the_same_counts(tmp_path):
    """Reproducibility is a claim the CLI makes; it is cheap to check."""
    first = parse_json(
        ok(aegisq("run", "ghz", "--qubits", 6, "--shots", 512, "--seed", 11, "--json"), "run")
    )
    second = parse_json(
        ok(aegisq("run", "ghz", "--qubits", 6, "--shots", 512, "--seed", 11, "--json"), "run")
    )
    assert first["counts"] == second["counts"]


def test_every_json_command_emits_only_json(identities, bundle, tmp_path):
    """`--json` means stdout is machine-readable, with nothing else on it.

    liboqs-python attaches a stdout log handler to its own logger when it
    is imported and immediately writes a line about faulthandler. Every
    command that touched the secure layer therefore emitted

        liboqs-python faulthandler is disabled
        { ... }

    which no JSON parser accepts -- `aegisq secure-run --json | jq` was
    simply broken. The banner is written during the import, to a handler
    holding the real `sys.stdout` object (`from sys import stdout`), so
    `redirect_stdout` cannot reach it; the import is wrapped in
    `logging.disable` instead.

    Checked across every `--json` command at once, because the next
    dependency to do this will not announce itself either.
    """
    ok(run_bundle(identities, bundle, tmp_path, name="json"), "secure-run")

    invocations = {
        "doctor": ("doctor", "--json"),
        "run": ("run", "ghz", "--qubits", 4, "--shots", 16, "--seed", 1, "--json"),
        "estimate": ("estimate", "--qubits", 20, "--ranks", 4, "--json"),
        "optimize": ("optimize", "qft", "--qubits", 10, "--ranks", 4, "--json"),
        "secure-inspect": ("secure-inspect", bundle, "--json"),
        "verify-result": (
            "verify-result",
            tmp_path / "json.result.json",
            "--cluster",
            identities["cluster_public"],
            "--json",
        ),
    }

    for name, argv in invocations.items():
        completed = ok(aegisq(*argv), name)
        try:
            json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"`aegisq {name} --json` emitted non-JSON on stdout: {exc}\n{completed.stdout[:200]!r}"
            )
