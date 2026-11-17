"""Our two optional changes to the authors' SwarmRaft code.

The authors' code lives unchanged in sim/swarmraft (git submodule of
kapeldev/SwarmRaft at commit 6e9d47d). With the defaults below, SwarmVariant
behaves exactly like the authors' SwarmVerification class:

  apply_noise=False  the authors' code draws range noise but stores the true
                     distance (Scaling_Swarm_Error.py, line 70). True keeps the
                     noisy distance, using the same random draws in the same order.
  tol=0.01           the pairwise vote tolerance tau in metres (authors' default).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "swarmraft"))
from Scaling_Swarm_Error import SwarmVerification  # noqa: E402  (authors' code)

AUTHORS_TOL_M = 0.01


class SwarmVariant(SwarmVerification):
    """The authors' verifier with switchable range noise and vote tolerance."""

    def __init__(self, n_drones, seed, distance_noise_std, apply_noise=False, tol=AUTHORS_TOL_M):
        super().__init__(n_drones=n_drones, seed=seed, distance_noise_std=distance_noise_std)
        self.apply_noise = apply_noise
        self.tol = tol

    def generate_distance_matrix(self):
        if not self.apply_noise:
            return super().generate_distance_matrix()
        n = self.n_drones
        measured = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    true_distance = np.linalg.norm(self.true_positions[i] - self.true_positions[j])
                    measured[i, j] = true_distance + np.random.normal(0, self.distance_noise_std)
        self.distance_matrix = measured
        return measured

    def verify_all_drones(self, **kwargs):
        return super().verify_all_drones(tol=self.tol, **kwargs)
