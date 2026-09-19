"""Shared pytest fixtures and availability guards for the AegisQ test suite."""

from __future__ import annotations

import pytest

import aegisq

#: Tolerance used when comparing amplitudes produced by different backends.
#: fp64 state-vector arithmetic accumulates error proportional to circuit
#: depth; 1e-10 is loose enough for the depths used in tests and tight enough
#: to catch a genuine kernel bug.
AMPLITUDE_TOL = 1e-10


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "native: requires the compiled C++ core")


@pytest.fixture(scope="session")
def tol() -> float:
    return AMPLITUDE_TOL


@pytest.fixture(scope="session")
def native_core():
    core = aegisq.native_core()
    if core is None:
        pytest.skip("native core is not built (run `make build`)")
    return core


def requires_qiskit():
    return pytest.mark.skipif(
        not _importable("qiskit"), reason="Qiskit is not installed (validation extra)"
    )


def requires_liboqs():
    return pytest.mark.skipif(
        not _importable("oqs"), reason="liboqs-python is not installed (crypto extra)"
    )


def random_circuit(
    num_qubits: int,
    depth: int,
    seed: int,
    *,
    two_qubit_prob: float = 0.35,
    name: str = "random",
):
    """Deterministic pseudo-random circuit over the supported gate set.

    Used by the cross-backend and Qiskit validation suites: the same seed must
    always produce the same instruction list so failures are reproducible.
    """
    import random as _random

    from aegisq.circuit import Circuit

    rng = _random.Random(seed)
    one_qubit = ["x", "y", "z", "h", "s", "t"]
    rotations = ["rx", "ry", "rz"]
    two_qubit = ["cx", "cz", "swap"]

    circuit = Circuit(num_qubits, name=f"{name}_n{num_qubits}_d{depth}_s{seed}")
    for _ in range(depth):
        for q in range(num_qubits):
            roll = rng.random()
            if roll < 0.45:
                getattr(circuit, rng.choice(one_qubit))(q)
            elif roll < 0.75:
                getattr(circuit, rng.choice(rotations))(q, rng.uniform(-3.14, 3.14))
        if num_qubits >= 2:
            pairs = list(range(num_qubits))
            rng.shuffle(pairs)
            for a, b in zip(pairs[::2], pairs[1::2], strict=False):
                if rng.random() < two_qubit_prob * 2:
                    getattr(circuit, rng.choice(two_qubit))(a, b)
    return circuit


def _importable(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False
