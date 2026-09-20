"""The A/A calibration: measuring the harness rather than the simulator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aegisq.benchmark.report import calibration_table


def _null(family: str, ranks: int, changes, base: float = 0.1) -> pd.DataFrame:
    """Null trials with prescribed apparent changes."""
    rows = []
    for trial, change in enumerate(changes):
        rows.append(
            {
                "circuit_family": family,
                "qubits": 20,
                "ranks": ranks,
                "trial": trial,
                "first_wall_s": base,
                "second_wall_s": base * (1 + change),
                "apparent_change": change,
            }
        )
    return pd.DataFrame(rows)


def test_resolution_is_the_quantile_its_p_bound_names():
    """The threshold and the claim it licenses have to be the same number.

    `resolution` is quoted as the level an effect must clear to carry
    `p <= 1/(n+1)`. If it were, say, the median, the bound would be
    decoration. With ten trials the quantile is 10/11, which on ten
    sorted samples lands on the largest.
    """
    changes = [0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07, -0.08, 0.09, -0.30]
    table = calibration_table(_null("ghz", 8, changes))

    assert len(table) == 1
    row = table.iloc[0]
    assert row["trials"] == 10
    assert row["p_bound"] == pytest.approx(1 / 11)
    assert row["resolution"] == pytest.approx(0.30, rel=1e-6)


def test_the_sign_of_a_null_change_is_discarded():
    """A slow second run and a slow first run are the same failure."""
    positive = calibration_table(_null("qft", 4, [0.2, 0.1, 0.05]))
    negative = calibration_table(_null("qft", 4, [-0.2, -0.1, -0.05]))

    assert positive["resolution"].iloc[0] == pytest.approx(negative["resolution"].iloc[0])


def test_more_launches_per_arm_cannot_worsen_the_resolution():
    """Minimum-over-launches is the estimator the projection describes.

    Interference on this machine is one-sided -- it makes a launch
    slower, never faster -- so taking the minimum over more independent
    launches moves the estimate towards the uncontended runtime. The
    projection must reflect that monotonically; if it did not, the
    `--launches` flag would be advertising an improvement the analysis
    does not believe in.
    """
    rng = np.random.default_rng(7)
    # A clean floor with an occasional slow launch: the shape measured.
    base = 0.1
    times = base * (
        1 + np.where(rng.random(40) < 0.25, rng.random(40) * 0.8, rng.random(40) * 0.05)
    )
    rows = []
    for trial in range(20):
        first, second = times[2 * trial], times[2 * trial + 1]
        rows.append(
            {
                "circuit_family": "ising",
                "qubits": 20,
                "ranks": 4,
                "trial": trial,
                "first_wall_s": first,
                "second_wall_s": second,
                "apparent_change": (second - first) / first,
            }
        )
    null = pd.DataFrame(rows)

    resolutions = [
        float(calibration_table(null, launches=k)["resolution"].iloc[0]) for k in (1, 2, 3, 5)
    ]
    assert resolutions == sorted(resolutions, reverse=True), resolutions
    assert resolutions[-1] < resolutions[0], "five launches should beat one"


def test_the_projection_is_reproducible():
    """A resampled figure that moves between runs would be unciteable."""
    null = _null("grover", 8, [0.05, -0.12, 0.2, -0.03, 0.08, -0.15])
    first = calibration_table(null, launches=3, seed=11)["resolution"].iloc[0]
    second = calibration_table(null, launches=3, seed=11)["resolution"].iloc[0]
    assert first == second


def test_an_empty_calibration_is_not_an_error():
    """No calibration data means the weaker floor applies, not a crash."""
    assert calibration_table(pd.DataFrame()).empty


def test_the_floor_follows_the_circuit_family():
    """A Grover comparison must be judged against Grover's own scatter.

    The point of measuring the null per family is that a GHZ chain
    finishing in 8 ms and a Grover circuit taking 450 ms are not
    equally timeable. A single machine-wide floor would either excuse
    GHZ's noise or discard Grover's real effects.
    """
    from aegisq.benchmark.report import mapping_table
    from tests.unit.test_benchmark import _mapping_raw

    raw = _mapping_raw(
        [
            ("ghz", "default", 1000, [1.00, 1.01]),
            ("ghz", "optimized", 1000, [0.95, 0.96]),
            ("qft", "default", 1000, [1.00, 1.01]),
            ("qft", "optimized", 500, [0.95, 0.96]),
        ]
    )
    # GHZ is noisy, QFT is quiet: the same 5% change should survive for
    # one and not the other.
    null = pd.concat(
        [
            _null("ghz", 4, [0.4, -0.3, 0.35, -0.38, 0.2]),
            _null("qft", 4, [0.01, -0.005, 0.008, -0.012, 0.004]),
        ],
        ignore_index=True,
    )
    table = mapping_table(raw, null).set_index("circuit_family")

    assert table.loc["ghz", "wall_noise_floor"] > table.loc["qft", "wall_noise_floor"]
    assert table.loc["qft", "wall_change_resolved"]
    assert not table.loc["ghz", "wall_change_resolved"]
