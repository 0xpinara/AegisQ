"""Phase 13: memory estimation and environment diagnostics."""

from __future__ import annotations

import json

import pytest

from aegisq.cli.main import main
from aegisq.runtime.hardware import format_bytes, memory_estimate


def test_single_process_estimate():
    estimate = memory_estimate(26, ranks=1, precision="fp64")
    assert estimate["total_bytes"] == 2**26 * 16  # 1 GiB
    assert estimate["bytes_per_rank"] == estimate["total_bytes"]
    assert estimate["local_qubits"] == 26
    assert estimate["global_qubits"] == 0


@pytest.mark.parametrize(
    "qubits, expected_gib",
    [(26, 1), (30, 16), (32, 64)],
)
def test_fp64_state_sizes_match_the_documented_table(qubits, expected_gib):
    assert memory_estimate(qubits)["total_bytes"] == expected_gib * 2**30


def test_forty_qubits_is_sixteen_tebibytes():
    assert memory_estimate(40)["total_bytes"] == 16 * 2**40


def test_ranks_divide_the_state():
    estimate = memory_estimate(30, ranks=8)
    assert estimate["local_qubits"] == 27
    assert estimate["global_qubits"] == 3
    assert estimate["bytes_per_rank"] * 8 == estimate["total_bytes"]


def test_fp32_halves_everything():
    big = memory_estimate(28, ranks=4, precision="fp64")
    small = memory_estimate(28, ranks=4, precision="fp32")
    assert small["total_bytes"] * 2 == big["total_bytes"]
    assert small["bytes_per_rank"] * 2 == big["bytes_per_rank"]


def test_peak_accounts_for_the_exchange_buffers():
    """The shard is not the whole story: exchanges need scratch space."""
    estimate = memory_estimate(28, ranks=4)
    shard = estimate["bytes_per_rank"]
    assert estimate["exchange_buffer_bytes"] == shard
    assert estimate["packing_buffer_bytes"] == shard // 2
    assert estimate["peak_bytes_per_rank"] == shard * 5 // 2


@pytest.mark.parametrize("ranks", [3, 5, 6, 12])
def test_non_power_of_two_rank_counts_are_rejected(ranks):
    with pytest.raises(ValueError, match="power of two"):
        memory_estimate(20, ranks=ranks)


def test_too_many_ranks_is_rejected_with_a_usable_limit():
    with pytest.raises(ValueError, match="at most 8 ranks"):
        memory_estimate(4, ranks=16)


def test_unknown_precision_is_rejected():
    with pytest.raises(ValueError, match="precision"):
        memory_estimate(10, precision="fp16")


@pytest.mark.parametrize(
    "value, expected",
    [
        (512, "512 B"),
        (2**10, "1.00 KiB"),
        (2**20, "1.00 MiB"),
        (2**30, "1.00 GiB"),
        (2**40, "1.00 TiB"),
    ],
)
def test_byte_formatting(value, expected):
    assert format_bytes(value) == expected


def test_cli_estimate_human_output(capsys):
    assert main(["estimate", "--qubits", "30", "--ranks", "4"]) == 0
    out = capsys.readouterr().out
    assert "16.00 GiB" in out  # total
    assert "4.00 GiB" in out  # per rank
    assert "local qubits:          28" in out
    assert "peak per rank" in out


def test_cli_estimate_json_output(capsys):
    assert main(["estimate", "--qubits", "24", "--ranks", "2", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["num_qubits"] == 24
    assert payload["ranks"] == 2
    assert payload["local_qubits"] == 23


def test_cli_estimate_rejects_impossible_configurations():
    with pytest.raises(SystemExit, match="power of two"):
        main(["estimate", "--qubits", "20", "--ranks", "6"])


def test_doctor_reports_the_mpi_library_when_available():
    from aegisq.runtime import hardware

    info = hardware.mpi_info()
    if info.extra.get("core_mpi"):
        assert info.extra["library"] not in (None, "none")
