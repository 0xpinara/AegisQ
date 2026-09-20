"""Measure what the post-quantum layer actually costs.

Research question 2 asks what overhead standardised post-quantum cryptography
adds to an HPC job. That splits into two measurements, both taken here:

* **primitive cost** — key generation, encapsulation, decapsulation, signing
  and verification for the ML-KEM and ML-DSA parameter sets, plus the exact
  key, ciphertext and signature sizes;
* **end-to-end cost** — the wall time to pack and to verify-and-open a real
  job bundle, and the bytes the envelope adds on top of the circuit.

The second is the one that matters for the claim: a few hundred microseconds
of lattice arithmetic is irrelevant next to a job that will move gigabytes
over MPI, and the measurement is here so that statement can be checked rather
than asserted.

Timing method
-------------
Each operation is run `iterations` times after a warm-up, and the **median**
is reported alongside the mean and standard deviation. The median resists the
occasional scheduling hiccup that a mean would absorb.
"""

from __future__ import annotations

import csv
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Parameter sets measured by default: the three ML-KEM and three ML-DSA
#: security levels, so the cost of choosing a stronger set is visible.
DEFAULT_KEMS = ("ML-KEM-512", "ML-KEM-768", "ML-KEM-1024")
DEFAULT_SIGNATURES = ("ML-DSA-44", "ML-DSA-65", "ML-DSA-87")

PQC_RAW_FIELDS = [
    "timestamp",
    "hostname",
    "cpu_model",
    "logical_cores",
    "os",
    "python_version",
    "liboqs_version",
    "aegisq_version",
    "git_commit",
    "experiment",
    "algorithm",
    "operation",
    "iterations",
    "median_us",
    "mean_us",
    "stdev_us",
    "min_us",
    "max_us",
    "bytes",
    "detail",
]


@dataclass
class Timing:
    """Summary statistics for one repeated operation."""

    operation: str
    samples: list[float]

    @property
    def median_us(self) -> float:
        return statistics.median(self.samples) * 1e6

    @property
    def mean_us(self) -> float:
        return statistics.fmean(self.samples) * 1e6

    @property
    def stdev_us(self) -> float:
        return statistics.stdev(self.samples) * 1e6 if len(self.samples) > 1 else 0.0

    @property
    def min_us(self) -> float:
        return min(self.samples) * 1e6

    @property
    def max_us(self) -> float:
        return max(self.samples) * 1e6


def time_operation(callable_: Callable[[], Any], iterations: int, warmup: int = 5) -> Timing:
    """Run an operation repeatedly and collect per-call timings."""
    for _ in range(warmup):
        callable_()
    samples: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        callable_()
        samples.append(time.perf_counter() - started)
    return Timing(operation="", samples=samples)


def _environment() -> dict[str, Any]:
    import platform
    import socket

    from aegisq import __version__
    from aegisq.benchmark.runner import git_commit
    from aegisq.runtime import hardware

    try:
        import oqs

        liboqs_version = oqs.oqs_version()
    except Exception:  # pragma: no cover - environment dependent
        liboqs_version = "unavailable"

    cpu = hardware.cpu_info()
    commit, _ = git_commit()
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": socket.gethostname(),
        "cpu_model": cpu.extra.get("model", "unknown"),
        "logical_cores": cpu.extra.get("logical_cores", 0),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "liboqs_version": liboqs_version,
        "aegisq_version": __version__,
        "git_commit": commit,
    }


def measure_kem(algorithm: str, iterations: int = 1000) -> list[dict[str, Any]]:
    """Key generation, encapsulation and decapsulation for one KEM."""
    import oqs

    environment = _environment()
    rows: list[dict[str, Any]] = []

    with oqs.KeyEncapsulation(algorithm) as instance:
        public_key = instance.generate_keypair()
        secret_key = instance.export_secret_key()
        details = dict(instance.details)

    def keygen() -> None:
        with oqs.KeyEncapsulation(algorithm) as handle:
            handle.generate_keypair()

    def encapsulate() -> None:
        with oqs.KeyEncapsulation(algorithm) as handle:
            handle.encap_secret(public_key)

    with oqs.KeyEncapsulation(algorithm) as encapsulator:
        ciphertext, _ = encapsulator.encap_secret(public_key)

    def decapsulate() -> None:
        with oqs.KeyEncapsulation(algorithm, secret_key) as handle:
            handle.decap_secret(ciphertext)

    for operation, function, size in (
        ("keygen", keygen, details["length_public_key"]),
        ("encapsulate", encapsulate, details["length_ciphertext"]),
        ("decapsulate", decapsulate, details["length_shared_secret"]),
    ):
        timing = time_operation(function, iterations)
        rows.append(
            {
                **environment,
                "experiment": "pqc_primitive",
                "algorithm": algorithm,
                "operation": operation,
                "iterations": iterations,
                "median_us": timing.median_us,
                "mean_us": timing.mean_us,
                "stdev_us": timing.stdev_us,
                "min_us": timing.min_us,
                "max_us": timing.max_us,
                "bytes": size,
                "detail": f"secret key {details['length_secret_key']} B",
            }
        )
    return rows


def measure_signature(algorithm: str, iterations: int = 1000) -> list[dict[str, Any]]:
    """Key generation, signing and verification for one signature scheme."""
    import oqs

    environment = _environment()
    message = b"aegisq benchmark message" * 8  # ~200 bytes, envelope-sized
    rows: list[dict[str, Any]] = []

    with oqs.Signature(algorithm) as instance:
        public_key = instance.generate_keypair()
        secret_key = instance.export_secret_key()
        signature = instance.sign(message)
        details = dict(instance.details)

    def keygen() -> None:
        with oqs.Signature(algorithm) as handle:
            handle.generate_keypair()

    def sign() -> None:
        with oqs.Signature(algorithm, secret_key) as handle:
            handle.sign(message)

    def verify() -> None:
        with oqs.Signature(algorithm) as handle:
            handle.verify(message, signature, public_key)

    for operation, function, size in (
        ("keygen", keygen, details["length_public_key"]),
        ("sign", sign, details["length_signature"]),
        ("verify", verify, len(signature)),
    ):
        timing = time_operation(function, iterations)
        rows.append(
            {
                **environment,
                "experiment": "pqc_primitive",
                "algorithm": algorithm,
                "operation": operation,
                "iterations": iterations,
                "median_us": timing.median_us,
                "mean_us": timing.mean_us,
                "stdev_us": timing.stdev_us,
                "min_us": timing.min_us,
                "max_us": timing.max_us,
                "bytes": size,
                "detail": f"secret key {details['length_secret_key']} B",
            }
        )
    return rows


def measure_envelope(qubits: int = 20, family: str = "qft", iterations: int = 50) -> list[dict]:
    """End-to-end cost of packing and opening a real job bundle."""
    import tempfile

    from aegisq.algorithms import build_circuit
    from aegisq.secure.canonical import canonical_bytes
    from aegisq.secure.envelope import ExecutionRequest, open_job, pack_job
    from aegisq.secure.keys import generate_client_identity, generate_cluster_identity

    environment = _environment()
    circuit = build_circuit(family, qubits)
    request = ExecutionRequest(ranks=8, shots=1024)

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        client = generate_client_identity("bench-client", root / "client")
        cluster = generate_cluster_identity("bench-cluster", root / "cluster")
        trusted = {client.public.signature_fingerprint: client.public}

        bundle = pack_job(circuit, request, client, cluster.public)
        plaintext_bytes = len(canonical_bytes(circuit.to_dict()))

        pack_timing = time_operation(
            lambda: pack_job(circuit, request, client, cluster.public), iterations, warmup=2
        )
        open_timing = time_operation(
            lambda: open_job(bundle, cluster, trusted), iterations, warmup=2
        )

    detail = f"{family} {qubits} qubits, {len(circuit)} gates"
    rows = []
    for operation, timing, size in (
        ("pack_job", pack_timing, len(bundle)),
        ("verify_and_open", open_timing, len(bundle)),
    ):
        rows.append(
            {
                **environment,
                "experiment": "pqc_envelope",
                "algorithm": "ML-KEM-768 + ML-DSA-65 + AES-256-GCM",
                "operation": operation,
                "iterations": iterations,
                "median_us": timing.median_us,
                "mean_us": timing.mean_us,
                "stdev_us": timing.stdev_us,
                "min_us": timing.min_us,
                "max_us": timing.max_us,
                "bytes": size,
                "detail": detail,
            }
        )
    # Size accounting, separated so the fixed cryptographic cost is not
    # confused with base64 expansion of the payload itself.
    encoded_payload = -(-plaintext_bytes // 3) * 4  # ceil(n/3) * 4
    for operation, size, note in (
        ("bundle_size", len(bundle), "complete .aqjob file"),
        ("payload_plaintext", plaintext_bytes, "canonical circuit + manifest"),
        ("payload_base64", encoded_payload, "the same payload after base64 expansion"),
        (
            "envelope_fixed_overhead",
            len(bundle) - encoded_payload,
            "KEM ciphertext, signature, header and AEAD tag",
        ),
    ):
        rows.append(
            {
                **environment,
                "experiment": "pqc_envelope",
                "algorithm": "ML-KEM-768 + ML-DSA-65 + AES-256-GCM",
                "operation": operation,
                "iterations": 1,
                "median_us": 0.0,
                "mean_us": 0.0,
                "stdev_us": 0.0,
                "min_us": 0.0,
                "max_us": 0.0,
                "bytes": size,
                "detail": f"{detail}; {note}",
            }
        )
    return rows


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PQC_RAW_FIELDS)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in PQC_RAW_FIELDS})


def run_suite(
    output: Path,
    iterations: int = 1000,
    envelope_iterations: int = 50,
    kems: tuple[str, ...] = DEFAULT_KEMS,
    signatures: tuple[str, ...] = DEFAULT_SIGNATURES,
    envelope_qubits: int = 20,
    verbose: bool = True,
) -> Path:
    """Measure every primitive and the end-to-end envelope, appending raw rows."""
    rows: list[dict[str, Any]] = []
    for algorithm in kems:
        if verbose:
            print(f"  measuring {algorithm} ({iterations} iterations)")
        rows += measure_kem(algorithm, iterations)
    for algorithm in signatures:
        if verbose:
            print(f"  measuring {algorithm} ({iterations} iterations)")
        rows += measure_signature(algorithm, iterations)
    if verbose:
        print(f"  measuring end-to-end envelope ({envelope_iterations} iterations)")
    rows += measure_envelope(qubits=envelope_qubits, iterations=envelope_iterations)

    append_rows(output, rows)
    return output
