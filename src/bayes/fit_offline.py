from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.bayes.diagnostics import save_bayes_diagnostics
from src.bayes.posterior_predictive import posterior_predictive_checks
from src.common.time import utc_now_iso
from src.models.bayes_state_space_bt import BayesStateSpaceBTModel



def run_bayes_offline_fit(
    features_df: pd.DataFrame,
    feature_columns: list[str],
    artifacts_dir: str,
    process_variance: float = 0.08,
    prior_variance: float = 1.5,
    draws: int = 500,
) -> tuple[BayesStateSpaceBTModel, dict]:
    model = BayesStateSpaceBTModel(
        process_variance=process_variance,
        prior_variance=prior_variance,
        draws=draws,
    )
    model.fit_offline(features_df, feature_columns=feature_columns, n_passes=3)

    train = features_df[features_df["home_win"].notna()].copy()
    summary = model.predict_summary(train)
    y = train["home_win"].astype(int).to_numpy()
    p = summary.mean
    ppc = posterior_predictive_checks(y_true=y, p_mean=p)
    pit_vals = np.where(y == 1, 1 - p, p)

    diagnostics = model.diagnostics() | {"ppc": ppc, "fitted_at_utc": utc_now_iso()}
    diag_paths = save_bayes_diagnostics(
        diagnostics=diagnostics,
        pit_values=pit_vals,
        out_dir=str(Path(artifacts_dir) / "validation"),
        prefix="bayes_offline",
    )
    diagnostics["artifact_paths"] = diag_paths
    return model, diagnostics
