from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.utils import ensure_dir



def save_glm_diagnostics(
    df: pd.DataFrame,
    p_col: str,
    y_col: str,
    feature_cols: list[str],
    coefs: np.ndarray,
    out_dir: str,
    prefix: str = "glm",
) -> dict[str, str]:
    out = ensure_dir(Path(out_dir))
    y = df[y_col].to_numpy(dtype=float)
    p = np.clip(df[p_col].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    resid_dev = np.sign(y - p) * np.sqrt(2 * (y * np.log(np.clip(y / p, 1e-6, None)) + (1 - y) * np.log(np.clip((1 - y) / (1 - p), 1e-6, None))))
    resid_work = (y - p) / np.sqrt(p * (1 - p))

    paths: dict[str, str] = {}

    plt.figure(figsize=(7, 4))
    plt.scatter(p, resid_dev, s=10, alpha=0.5)
    plt.axhline(0, color="black", lw=1)
    plt.xlabel("Fitted probability")
    plt.ylabel("Deviance residual")
    plt.title("Deviance Residuals vs Fitted")
    plt.tight_layout()
    p1 = out / f"{prefix}_deviance_residuals.png"
    plt.savefig(p1, dpi=140)
    plt.close()
    paths["deviance"] = str(p1)

    plt.figure(figsize=(7, 4))
    plt.scatter(p, resid_work, s=10, alpha=0.5)
    plt.axhline(0, color="black", lw=1)
    plt.xlabel("Fitted probability")
    plt.ylabel("Working residual")
    plt.title("Working Residuals vs Fitted")
    plt.tight_layout()
    p2 = out / f"{prefix}_working_residuals.png"
    plt.savefig(p2, dpi=140)
    plt.close()
    paths["working"] = str(p2)

    top_ix = np.argsort(np.abs(coefs))[::-1][: min(5, len(feature_cols))]
    for i in top_ix:
        x = df[feature_cols[i]].to_numpy(dtype=float)
        partial = resid_work + coefs[i] * x
        plt.figure(figsize=(7, 4))
        plt.scatter(x, partial, s=10, alpha=0.5)
        plt.axhline(0, color="black", lw=1)
        plt.xlabel(feature_cols[i])
        plt.ylabel("Partial residual")
        plt.title(f"Partial Residual Plot: {feature_cols[i]}")
        plt.tight_layout()
        px = out / f"{prefix}_partial_{feature_cols[i]}.png"
        plt.savefig(px, dpi=140)
        plt.close()
        paths[f"partial_{feature_cols[i]}"] = str(px)

    return paths
