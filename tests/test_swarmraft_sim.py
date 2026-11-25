"""Checks that the wrapper reproduces the authors' code and that our switches work.

Run: .venv/bin/python -m pytest tests -q
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SIM = Path(__file__).resolve().parents[1] / "sim"
sys.path.insert(0, str(SIM))

from swarmraft_sim import parse_params, simulate  # noqa: E402
from swarmraft_variants import SwarmVariant  # noqa: E402  (puts the authors' code on sys.path)

from Scaling_Swarm_Error import SwarmVerification  # noqa: E402  (authors' code)


def authors_errors(n, f, runs, seed, noise=0.5):
    swarm = SwarmVerification(n_drones=n, seed=seed, distance_noise_std=noise)
    return [swarm.run_full_verification(num_faulty=f)["avg_error_verified"] for _ in range(runs)]


def test_defaults_reproduce_authors_code_exactly():
    ours = [r["err_recovered"] for r in simulate(parse_params({"n": 7, "f": 3, "runs": 20}), seed=43)]
    assert ours == authors_errors(7, 3, 20, seed=43)


def test_authors_code_ignores_noise_setting():
    assert authors_errors(5, 2, 20, seed=1, noise=0.0) == authors_errors(5, 2, 20, seed=1, noise=0.5)


def test_apply_noise_stores_noisy_ranges_with_same_geometry():
    def build(apply_noise):  # the authors' class seeds numpy's global generator, so build one at a time
        swarm = SwarmVariant(6, seed=3, distance_noise_std=0.5, apply_noise=apply_noise)
        swarm.generate_swarm(num_faulty=2)
        swarm.generate_distance_matrix()
        return swarm

    plain, noisy = build(False), build(True)
    np.testing.assert_array_equal(plain.true_positions, noisy.true_positions)
    diff = noisy.distance_matrix - plain.distance_matrix
    assert 0.3 < diff[~np.eye(6, dtype=bool)].std() < 0.8


def test_gnss_noise_moves_every_report_and_nothing_else():
    def build(gnss_noise):
        swarm = SwarmVariant(9, seed=3, distance_noise_std=0.5, gnss_noise=gnss_noise)
        swarm.generate_swarm(num_faulty=4)
        swarm.generate_distance_matrix()
        return swarm

    plain, noisy = build(0.0), build(1.0)
    np.testing.assert_array_equal(plain.true_positions, noisy.true_positions)
    np.testing.assert_array_equal(plain.faulty_indices, noisy.faulty_indices)
    np.testing.assert_array_equal(plain.distance_matrix, noisy.distance_matrix)
    diff = noisy.reported_positions - plain.reported_positions
    assert (diff != 0).all()  # every drone, every axis
    assert 0.6 < diff.std() < 1.4


def test_tolerance_is_passed_to_the_vote():
    rows = simulate(parse_params({"n": 9, "f": 4, "runs": 30, "apply_noise": True}), seed=5)
    rows_tol = simulate(parse_params({"n": 9, "f": 4, "runs": 30, "apply_noise": True, "tol": 2.0}), seed=5)
    moved = sum(r["honest_moved"] for r in rows)
    moved_tol = sum(r["honest_moved"] for r in rows_tol)
    assert moved > moved_tol


def test_noise_free_run_never_moves_honest_drones():
    rows = simulate(parse_params({"n": 9, "f": 4, "runs": 50}), seed=43)
    assert all(r["honest_moved"] == 0 and r["err_honest"] == 0 for r in rows)


def test_scenario_mode_returns_one_row_per_drone():
    rows = simulate(parse_params({"n": 6, "f": 2, "mode": "scenario"}), seed=42)
    df = pd.DataFrame(rows)
    assert len(df) == 6 and df.spoofed.sum() == 2
    assert {"true_x", "reported_z", "recovered_y", "flagged"} <= set(df.columns)


@pytest.mark.parametrize("raw", [{"n": 5}, {"n": 4, "f": 2}, {"n": 5, "f": 1, "foo": 1},
                                 {"n": 5, "f": 1, "mode": "x"}, {"n": 5, "f": -1}])
def test_invalid_parameters_are_rejected(raw):
    with pytest.raises(ValueError):
        parse_params(raw)


def test_command_line_contract(tmp_path):
    out = tmp_path / "r.csv"
    subprocess.run([sys.executable, "swarmraft_sim.py", "--params", json.dumps({"n": 5, "f": 2, "runs": 3}),
                    "--seed", "43", "--out", str(out)], cwd=SIM, check=True,
                   env={**os.environ, "MPLBACKEND": "Agg"})
    df = pd.read_csv(out)
    assert list(df.run) == [0, 1, 2]
    assert {"err_recovered", "err_spoofed", "honest_moved", "spoofed_missed"} <= set(df.columns)
