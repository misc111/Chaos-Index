from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from src.common.utils import ensure_dir



def permutation_importance_report(
    model,
    x_df: pd.DataFrame,
    y,
    out_dir: str,
    model_name: str,
    n_repeats: int = 20,
) -> pd.DataFrame:
    out = ensure_dir(Path(out_dir))
    imp = permutation_importance(model, x_df, y, n_repeats=n_repeats, random_state=42, scoring="neg_log_loss")
    res = pd.DataFrame(
        {
            "feature": x_df.columns,
            "importance_mean": imp.importances_mean,
            "importance_std": imp.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    csv_path = out / f"{model_name}_permutation_importance.csv"
    res.to_csv(csv_path, index=False)

    top = res.head(20).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    plt.title(f"Permutation Importance ({model_name})")
    plt.xlabel("Mean importance (neg log loss)")
    plt.tight_layout()
    plt.savefig(out / f"{model_name}_permutation_importance.png", dpi=140)
    plt.close()

    return res
