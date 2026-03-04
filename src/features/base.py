from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class FeatureBuildResult:
    dataframe: pd.DataFrame
    feature_columns: list[str]
    feature_set_version: str
    metadata: dict[str, Any]
