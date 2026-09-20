"""Phase 20: the post-quantum measurement harness.

The measurements themselves are timings and cannot be asserted, but their
*shape* can: the right operations are measured, sizes match the published
parameter sets, and the size accounting adds up.
"""

from __future__ import annotations

import csv

import pytest

from aegisq.benchmark.pqc import (
    PQC_RAW_FIELDS,
    append_rows,
    measure_envelope,
    measure_kem,
    measure_signature,
    time_operation,
)
from tests.conftest import requires_liboqs

pytestmark = [pytest.mark.crypto, requires_liboqs()]


def test_timing_collects_one_sample_per_iteration():
    timing = time_operation(lambda: sum(range(100)), iterations=20, warmup=2)
    assert len(timing.samples) == 20
    assert timing.median_us > 0
    assert timing.min_us <= timing.median_us <= timing.max_us


def test_single_sample_has_zero_spread():
    timing = time_operation(lambda: None, iterations=1, warmup=0)
    assert timing.stdev_us == 0.0


@pytest.mark.parametrize(
    "algorithm, public_key_bytes, ciphertext_bytes",
    [("ML-KEM-512", 800, 768), ("ML-KEM-768", 1184, 1088), ("ML-KEM-1024", 1568, 1568)],
)
def test_kem_sizes_match_the_specification(algorithm, public_key_bytes, ciphertext_bytes):
    rows = {row["operation"]: row for row in measure_kem(algorithm, iterations=5)}
    assert set(rows) == {"keygen", "encapsulate", "decapsulate"}
    assert rows["keygen"]["bytes"] == public_key_bytes
    assert rows["encapsulate"]["bytes"] == ciphertext_bytes
    assert rows["decapsulate"]["bytes"] == 32  # shared secret
    for row in rows.values():
        assert row["median_us"] > 0
        assert row["algorithm"] == algorithm


@pytest.mark.parametrize(
    "algorithm, public_key_bytes, signature_bytes",
    [("ML-DSA-44", 1312, 2420), ("ML-DSA-65", 1952, 3309), ("ML-DSA-87", 2592, 4627)],
)
def test_signature_sizes_match_the_specification(algorithm, public_key_bytes, signature_bytes):
    rows = {row["operation"]: row for row in measure_signature(algorithm, iterations=5)}
    assert set(rows) == {"keygen", "sign", "verify"}
    assert rows["keygen"]["bytes"] == public_key_bytes
    assert rows["sign"]["bytes"] == signature_bytes


def test_stronger_parameter_sets_are_not_cheaper():
    """Sanity check on the measurement, not a claim about implementations."""
    sizes = {}
    for algorithm in ("ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"):
        rows = {row["operation"]: row for row in measure_kem(algorithm, iterations=5)}
        sizes[algorithm] = rows["encapsulate"]["bytes"]
    assert sizes["ML-KEM-512"] < sizes["ML-KEM-768"] < sizes["ML-KEM-1024"]


def test_envelope_measurement_decomposes_the_size():
    rows = {row["operation"]: row for row in measure_envelope(qubits=8, iterations=3)}
    assert {
        "pack_job",
        "verify_and_open",
        "bundle_size",
        "payload_plaintext",
        "payload_base64",
        "envelope_fixed_overhead",
    } <= set(rows)

    bundle = rows["bundle_size"]["bytes"]
    encoded = rows["payload_base64"]["bytes"]
    plaintext = rows["payload_plaintext"]["bytes"]
    overhead = rows["envelope_fixed_overhead"]["bytes"]

    assert encoded >= plaintext  # base64 expands
    assert overhead == bundle - encoded
    # The fixed part must at least cover a KEM ciphertext and a signature.
    assert overhead > 1088 + 3309


def test_envelope_overhead_is_independent_of_circuit_size():
    small = {r["operation"]: r for r in measure_envelope(qubits=6, iterations=2)}
    large = {r["operation"]: r for r in measure_envelope(qubits=12, iterations=2)}
    assert large["payload_plaintext"]["bytes"] > small["payload_plaintext"]["bytes"]
    # Fixed overhead varies only by a few bytes of length fields and padding.
    difference = abs(
        large["envelope_fixed_overhead"]["bytes"] - small["envelope_fixed_overhead"]["bytes"]
    )
    assert difference < 64


def test_rows_carry_provenance_and_write_to_csv(tmp_path):
    rows = measure_kem("ML-KEM-768", iterations=3)
    path = tmp_path / "pqc.csv"
    append_rows(path, rows)
    append_rows(path, rows)

    with path.open(encoding="utf-8") as handle:
        contents = list(csv.reader(handle))
    assert contents[0] == PQC_RAW_FIELDS
    assert len(contents) == 1 + 2 * len(rows)
    for row in rows:
        assert row["liboqs_version"]
        assert row["cpu_model"]
        assert row["git_commit"]
