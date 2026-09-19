"""Phase 4: partitioning arithmetic.

These tests run in a single process. `DistributedLayout` takes the world size
and rank explicitly, so every placement rule can be checked for ranks that do
not exist locally — no mpirun required.
"""

from __future__ import annotations

import pytest

from aegisq.runtime.distributed import layout

pytestmark = pytest.mark.native


@pytest.mark.parametrize(
    "num_qubits, world_size, expected_local, expected_global",
    [
        (6, 1, 6, 0),
        (6, 2, 5, 1),
        (6, 4, 4, 2),
        (6, 8, 3, 3),
        (24, 8, 21, 3),
    ],
)
def test_local_and_global_split(num_qubits, world_size, expected_local, expected_global):
    lay = layout(num_qubits, world_size, 0)
    assert lay.num_local_qubits == expected_local
    assert lay.num_global_qubits == expected_global
    assert lay.local_state_size == 2**expected_local


def test_local_state_bytes_matches_precision():
    lay = layout(30, 4, 0)
    assert lay.local_state_bytes(16) == 2**28 * 16  # fp64 complex
    assert lay.local_state_bytes(8) == 2**28 * 8  # fp32 complex


def test_default_mapping_makes_the_highest_qubits_global():
    lay = layout(6, 4, 0)
    assert lay.global_qubits() == [4, 5]
    assert lay.local_qubits() == [0, 1, 2, 3]
    assert lay.is_identity_mapping()
    for q in range(4):
        assert lay.is_local(q)
    for q in (4, 5):
        assert lay.is_global(q)


@pytest.mark.parametrize("rank", range(8))
def test_global_bit_is_read_from_the_rank_id(rank):
    lay = layout(7, 8, rank)
    # Global qubits 4, 5, 6 occupy global positions 0, 1, 2.
    assert lay.global_bit(4) == (rank >> 0) & 1
    assert lay.global_bit(5) == (rank >> 1) & 1
    assert lay.global_bit(6) == (rank >> 2) & 1


@pytest.mark.parametrize("rank", range(8))
def test_partner_rank_flips_exactly_one_bit(rank):
    lay = layout(7, 8, rank)
    for qubit, position in ((4, 0), (5, 1), (6, 2)):
        partner = lay.partner_rank_for_global_qubit(qubit)
        assert partner == rank ^ (1 << position)
        assert lay.partner_rank_of(qubit, partner) == rank  # involution


def test_physical_index_composition():
    lay = layout(5, 4, 2)  # L = 3
    assert lay.physical_index(0) == 2 * 8
    assert lay.physical_index(5) == 2 * 8 + 5
    assert lay.physical_index_for_rank(3, 1) == 3 * 8 + 1


def test_identity_mapping_keeps_logical_and_physical_indices_equal():
    lay = layout(5, 4, 0)
    for index in range(32):
        assert lay.to_logical_index(index) == index
        assert lay.to_physical_index(index) == index


def test_custom_mapping_permutes_positions():
    # Put logical qubits 1 and 5 on the two global positions (4 and 5).
    mapping = [0, 4, 1, 2, 3, 5]
    lay = layout(6, 4, 1, mapping)
    assert lay.global_qubits() == [1, 5]
    assert sorted(lay.local_qubits()) == [0, 2, 3, 4]
    assert not lay.is_identity_mapping()
    assert lay.is_global(1) and lay.is_global(5)
    assert lay.is_local(0) and lay.is_local(4)


def test_custom_mapping_index_translation_is_a_bijection():
    mapping = [2, 5, 0, 4, 1, 3]
    lay = layout(6, 4, 0, mapping)
    seen = set()
    for logical in range(64):
        physical = lay.to_physical_index(logical)
        assert lay.to_logical_index(physical) == logical
        seen.add(physical)
    assert len(seen) == 64


def test_mapping_moves_the_expected_bit():
    # logical qubit 0 -> position 5 (a global position for 4 ranks over 6 qubits)
    mapping = [5, 0, 1, 2, 3, 4]
    lay = layout(6, 4, 0, mapping)
    assert lay.position(0) == 5
    assert lay.to_physical_index(0b000001) == 0b100000


@pytest.mark.parametrize("world_size", [3, 5, 6, 12])
def test_non_power_of_two_world_is_rejected(world_size):
    with pytest.raises(Exception, match="power of two"):
        layout(8, world_size, 0)


def test_too_few_qubits_for_the_rank_count_is_rejected():
    with pytest.raises(Exception, match="at least one local qubit"):
        layout(2, 4, 0)


def test_rank_must_be_inside_the_world():
    with pytest.raises(Exception, match="rank out of range"):
        layout(6, 4, 4)


@pytest.mark.parametrize("mapping", [[0, 1, 2], [0, 0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 9]])
def test_invalid_mappings_are_rejected(mapping):
    with pytest.raises(Exception, match="mapping"):
        layout(6, 4, 0, mapping)


def test_asking_for_the_global_position_of_a_local_qubit_fails():
    lay = layout(6, 4, 0)
    with pytest.raises(Exception, match="not global"):
        lay.global_position(0)
