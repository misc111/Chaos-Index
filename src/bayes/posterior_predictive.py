from __future__ import annotations

import numpy as np
import pandas as pd



def posterior_predictive_checks(y_true: np.ndarray, p_mean: np.ndarray, draws: int = 500, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(p_mean, dtype=float), 1e-6, 1 - 1e-6)

    sim_means = []
    for _ in range(draws):
        y_sim = rng.binomial(1, p)
        sim_means.append(float(y_sim.mean()))

    obs_mean = float(y.mean()) if y.size else 0.0
    pit = np.where(y == 1, 1 - p, p)

    return {
        "observed_mean": obs_mean,
        "simulated_mean_avg": float(np.mean(sim_means)) if sim_means else 0.0,
        "simulated_mean_std": float(np.std(sim_means)) if sim_means else 0.0,
        "pit_mean": float(np.mean(pit)) if pit.size else 0.0,
        "pit_std": float(np.std(pit)) if pit.size else 0.0,
    }



def pit_histogram(pit_values: np.ndarray, bins: int = 10) -> pd.DataFrame:
    vals = np.asarray(pit_values, dtype=float)
    hist, edges = np.histogram(vals, bins=bins, range=(0, 1), density=False)
    return pd.DataFrame(
        {
            "bin_left": edges[:-1],
            "bin_right": edges[1:],
            "count": hist,
        }
    )
