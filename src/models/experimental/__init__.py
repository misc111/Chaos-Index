"""Experimental challenger model implementations.

These models are intentionally kept out of the CAS-governed theory-core lane.
Training and tests should import challengers from this namespace rather than
from the top-level ``src.models`` package.
"""

from src.models.experimental.bayes_state_space_bt import BayesStateSpaceBTModel, BayesSummary
from src.models.experimental.bayes_state_space_goals import BayesGoalsModel
from src.models.experimental.gbdt import GBDTModel
from src.models.experimental.nn import NNModel
from src.models.experimental.rf import RFModel

__all__ = [
    "BayesGoalsModel",
    "BayesStateSpaceBTModel",
    "BayesSummary",
    "GBDTModel",
    "NNModel",
    "RFModel",
]
