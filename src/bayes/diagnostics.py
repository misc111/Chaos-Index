from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.bayes.posterior_predictive import pit_histogram
from src.common.utils import ensure_dir



def save_bayes_diagnostics(
    diagnostics: dict,
    pit_values: np.ndarray,
    out_dir: str,
    prefix: str = "bayes",
) -> dict[str, str]:
    out = ensure_dir(Path(out_dir))
    paths: dict[str, str] = {}

    diag_json = out / f"{prefix}_diagnostics.json"
    diag_json.write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
    paths["diagnostics_json"] = str(diag_json)

    elbo = diagnostics.get("elbo_trace", [])
    if elbo:
        plt.figure(figsize=(7, 4))
        plt.plot(elbo, marker="o")
        plt.title("Bayes VI/Filter Objective Trace")
        plt.xlabel("Pass")
        plt.ylabel("Objective (higher better)")
        plt.tight_layout()
        elbo_path = out / f"{prefix}_elbo_trace.png"
        plt.savefig(elbo_path, dpi=140)
        plt.close()
        paths["elbo_plot"] = str(elbo_path)

    pit_df = pit_histogram(pit_values, bins=10)
    pit_csv = out / f"{prefix}_pit_histogram.csv"
    pit_df.to_csv(pit_csv, index=False)
    paths["pit_csv"] = str(pit_csv)

    plt.figure(figsize=(7, 4))
    plt.bar((pit_df["bin_left"] + pit_df["bin_right"]) / 2, pit_df["count"], width=0.09)
    plt.title("PIT-style Calibration Histogram")
    plt.xlabel("PIT bin")
    plt.ylabel("count")
    plt.tight_layout()
    pit_plot = out / f"{prefix}_pit_histogram.png"
    plt.savefig(pit_plot, dpi=140)
    plt.close()
    paths["pit_plot"] = str(pit_plot)

    return paths
