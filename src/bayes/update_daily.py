from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.time import utc_now_iso
from src.models.bayes_state_space_bt import BayesStateSpaceBTModel



def run_bayes_daily_update(
    model_path: str,
    new_results_features: pd.DataFrame,
) -> dict:
    model = BayesStateSpaceBTModel.load(model_path)
    model.daily_update(new_results_features)
    model.save(model_path)
    return {
        "updated_at_utc": utc_now_iso(),
        "model_path": str(Path(model_path)),
        "n_games": int(len(new_results_features)),
    }
