"""Run the authors' SwarmRaft simulation with the template's command-line contract.

    python swarmraft_sim.py --params '<json>' --seed <int> --out <csv>

Parameters (JSON):
  n            swarm size
  f            number of spoofed drones
  runs         Monte Carlo runs in this job (default 1000)
  noise        range noise std in metres, as passed to the authors' code (default 0.5)
  apply_noise  keep the noisy ranges instead of the true ones (default false = authors)
  tol          pairwise vote tolerance in metres (default 0.01 = authors)
  gnss_noise   GNSS noise std per axis in metres on every reported position (default 0 = authors)
  mode         "montecarlo" (default): one row per run
               "scenario": one row per drone of a single run (for Fig. 1)

The seed is passed to the authors' class, which seeds numpy's global generator
once; Fig. 1 uses seed 42 and Fig. 2 uses seed 43 in the authors' scripts.
"""

import argparse
import json

import numpy as np
import pandas as pd

from swarmraft_variants import AUTHORS_TOL_M, SwarmVariant

DEFAULTS = {"runs": 1000, "noise": 0.5, "apply_noise": False, "tol": AUTHORS_TOL_M, "gnss_noise": 0.0,
            "mode": "montecarlo"}
FLOAT_FORMAT = "%.6g"


def parse_params(raw: dict) -> dict:
    unknown = set(raw) - set(DEFAULTS) - {"n", "f"}
    if unknown:
        raise ValueError(f"unknown parameters: {sorted(unknown)}")
    p = {**DEFAULTS, **raw}
    if "n" not in p or "f" not in p:
        raise ValueError("parameters 'n' and 'f' are required")
    if not (p["f"] >= 0 and p["n"] >= 2 * p["f"] + 1):
        raise ValueError(f"need n >= 2f + 1, got n={p['n']}, f={p['f']}")
    if p["mode"] not in ("montecarlo", "scenario"):
        raise ValueError(f"unknown mode {p['mode']!r}")
    return p


def run_row(result: dict, run: int) -> dict:
    """Errors and detection outcome of one run (the authors' result dict)."""
    true, final = result["true_positions"], result["verification"]["verified_positions"]
    flags = result["verification"]["fault_flags"]
    spoofed = np.zeros(len(true), dtype=bool)
    spoofed[result["faulty_indices"]] = True
    honest_err = np.linalg.norm(final[~spoofed] - true[~spoofed], axis=1)
    return {
        "run": run,
        "err_recovered": result["avg_error_verified"],  # authors' metric: spoofed drones only
        "err_spoofed": result["avg_error_reported"],
        "err_honest": float(honest_err.mean()),
        "honest_moved": int(np.sum(flags & ~spoofed)),
        "spoofed_missed": int(np.sum(~flags & spoofed)),
    }


def scenario_rows(result: dict) -> list[dict]:
    """Positions of every drone in one run (true, reported, after verification)."""
    flags = result["verification"]["fault_flags"]
    spoofed = set(int(i) for i in result["faulty_indices"])
    rows = []
    for i, (t, r, v) in enumerate(zip(result["true_positions"], result["reported_positions"],
                                      result["verification"]["verified_positions"])):
        rows.append({"drone": i, "spoofed": i in spoofed, "flagged": bool(flags[i]),
                     **{f"true_{a}": x for a, x in zip("xyz", t)},
                     **{f"reported_{a}": x for a, x in zip("xyz", r)},
                     **{f"recovered_{a}": x for a, x in zip("xyz", v)}})
    return rows


def simulate(p: dict, seed: int) -> list[dict]:
    swarm = SwarmVariant(n_drones=p["n"], seed=seed, distance_noise_std=p["noise"],
                         apply_noise=p["apply_noise"], tol=p["tol"], gnss_noise=p["gnss_noise"])
    if p["mode"] == "scenario":
        return scenario_rows(swarm.run_full_verification(num_faulty=p["f"]))
    return [run_row(swarm.run_full_verification(num_faulty=p["f"]), k) for k in range(p["runs"])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", type=json.loads, required=True, help="JSON dict, see module docstring")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True, help="output CSV path")
    a = ap.parse_args()
    rows = simulate(parse_params(a.params), a.seed)
    pd.DataFrame(rows).to_csv(a.out, index=False, float_format=FLOAT_FORMAT)


if __name__ == "__main__":
    main()
