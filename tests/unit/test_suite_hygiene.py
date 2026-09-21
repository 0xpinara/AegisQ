"""Checks on the test suite itself.

A test suite is a measuring instrument, and these are its calibration
marks. Both properties below were broken at the same time, and neither
broke a test: one silently imported another project's fixtures, the other
silently ran nothing at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_tests_package_resolves_inside_this_repository():
    """`tests` must be this repository's package, not whichever one is first.

    Every module here imports its fixtures as `from tests.conftest import
    ...`. Without `__init__.py`, `tests` is only a namespace *portion*: the
    import machinery notes the directory and keeps scanning `sys.path`, and
    the first real `tests/__init__.py` anywhere on the path wins outright.

    On a machine with an unrelated editable install, that is exactly what
    happened -- `easy-install.pth` put a sibling checkout on the path, its
    `tests` was a regular package, and this suite's imports resolved into
    it. The names happened not to match, so it surfaced as a collection
    error. Had they matched, the suite would have passed against a
    stranger's fixtures.
    """
    import tests

    assert Path(tests.__file__).resolve().parent == REPO_ROOT / "tests"


@pytest.mark.parametrize(
    "package",
    ["tests", "tests.unit", "tests.integration", "tests.mpi", "tests.crypto"],
)
def test_every_test_directory_is_a_real_package(package):
    """One missing `__init__.py` reopens the hole for that directory alone."""
    module = __import__(package, fromlist=["__file__"])
    assert module.__file__ is not None, f"{package} is a namespace package"
    assert Path(module.__file__).name == "__init__.py"


def test_missing_mpi_support_fails_under_a_launcher(monkeypatch):
    """A launcher was used, so "nothing to test" is a failure, not a skip.

    Skipping is right when someone runs `pytest tests/mpi` on a build
    without MPI. Under `mpirun` it means the distributed suite proved
    nothing while exiting zero -- 166 skips that read exactly like 166
    passes.
    """
    from aegisq.runtime import distributed
    from tests.mpi import conftest

    monkeypatch.setattr(distributed, "mpi_compiled", lambda: False)
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "4")

    with pytest.raises(BaseException) as caught:
        conftest._require_mpi_core()
    assert "launched by an MPI launcher" in str(caught.value)


def test_missing_mpi_support_skips_without_a_launcher(monkeypatch):
    """Outside a launcher there is genuinely nothing to run."""
    from aegisq.runtime import distributed
    from tests.mpi import conftest

    monkeypatch.setattr(distributed, "mpi_compiled", lambda: False)
    for name in conftest._LAUNCHER_VARS:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(BaseException) as caught:
        conftest._require_mpi_core()
    assert type(caught.value).__name__ == "Skipped", f"expected a skip, got {caught.value!r}"


def test_naming_the_secret_half_never_loads_secret_material(tmp_path):
    """Asking for a public key must not read the secret file, ever.

    The identity resolver accepts either half's filename so that
    `--cluster hpc1`, `--cluster hpc1.public.json` and
    `--cluster hpc1.secret.json` all work -- the same flag used to mean a
    different one of the three in each subcommand. Convenience here has
    an obvious failure mode, so it is pinned: naming the secret half
    resolves to the public one rather than parsing secret material as a
    public key.
    """
    pytest.importorskip("oqs", reason="liboqs-python is not installed (crypto extra)")
    from aegisq.secure.keys import generate_cluster_identity, resolve_identity_file

    generate_cluster_identity("hpc1", tmp_path)
    public = tmp_path / "hpc1.public.json"
    secret = tmp_path / "hpc1.secret.json"

    for spelling in (tmp_path / "hpc1", public, secret):
        assert resolve_identity_file(spelling, "public") == public
    for spelling in (tmp_path / "hpc1", public, secret):
        assert resolve_identity_file(spelling, "secret") == secret


def test_a_missing_identity_reports_what_it_looked_for(tmp_path):
    """The old message named a path the user had not typed."""
    from aegisq.secure.keys import IdentityError, resolve_identity_file

    with pytest.raises(IdentityError) as caught:
        resolve_identity_file(tmp_path / "absent", "public")
    message = str(caught.value)
    assert "absent.public.json" in message and "tried" in message


def test_the_documented_gate_count_matches_the_parser():
    """ "Twelve gates" is a claim about the input language, and it drifted.

    Adding the fused `u` opcode took the IR to thirteen while three
    places still said twelve. The number that belongs in prose is the
    one the OpenQASM reader accepts, since `u` is an internal product
    the reader rejects on purpose.
    """
    from aegisq.circuit.gates import GATE_SPECS
    from aegisq.circuit.qasm import QASM_OPCODES

    assert set(GATE_SPECS) - set(QASM_OPCODES) == {"u"}
    assert len(QASM_OPCODES) == 12, (
        f"the input language now has {len(QASM_OPCODES)} gates; "
        "README.md and paper/main.tex say twelve"
    )

    root = Path(__file__).resolve().parents[2]
    for name in ("README.md", "paper/main.tex"):
        text = (root / name).read_text(encoding="utf-8")
        assert "twelve" in text, f"{name} no longer states the gate count"


def test_the_qasm_reader_refuses_the_fused_gate():
    """It is not part of the documented subset, and says so by name."""
    import pytest as _pytest

    from aegisq.circuit.qasm import QasmError, parse_qasm

    source = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nu(1,0,0,0,0,0,0,1) q[0];\n'
    with _pytest.raises(QasmError, match="internal representation"):
        parse_qasm(source)


def test_every_repository_url_matches_the_real_remote():
    """The first command in the README has to be the one that works.

    It was not. The quick start cloned `0xpinara/AegisQ-HPC`, which is a
    404, then changed into a directory that would not exist, and
    pyproject pointed package metadata at the same missing repository.
    Checked against the configured git remote rather than the network,
    so this runs offline and in CI.
    """
    import re
    import subprocess

    root = Path(__file__).resolve().parents[2]
    remote = subprocess.run(
        ["git", "remote", "get-url", "origin"],
        capture_output=True,
        text=True,
        cwd=root,
        check=False,
    )
    if remote.returncode != 0 or not remote.stdout.strip():
        pytest.skip("no git remote configured")

    slug = re.sub(r"\.git$", "", remote.stdout.strip()).rsplit("/", 2)[-2:]
    expected = "/".join(slug)

    pattern = re.compile(r"github\.com[:/]([\w.-]+/[\w.-]+?)(?:\.git)?(?=[)\s\"'/]|$)")
    for name in ("README.md", "pyproject.toml", "CONTRIBUTING.md"):
        path = root / name
        if not path.exists():
            continue
        for found in set(pattern.findall(path.read_text(encoding="utf-8"))):
            if found.split("/")[0] != expected.split("/")[0]:
                continue  # a third party's repository, not ours
            assert found == expected, f"{name} points at {found}, remote is {expected}"
