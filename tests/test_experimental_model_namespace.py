from src.models.bayes_state_space_goals import BayesGoalsModel as LegacyBayesGoalsModel
from src.models.bayes_state_space_bt import BayesStateSpaceBTModel as LegacyBayesBTModel
from src.models.experimental import BayesGoalsModel, BayesStateSpaceBTModel, GBDTModel, NNModel, RFModel
from src.models.experimental.bayes_state_space_goals import BayesGoalsModel as ModuleBayesGoalsModel
from src.models.experimental.gbdt import GBDTModel as ModuleGBDTModel
from src.models.experimental.nn import NNModel as ModuleNNModel
from src.models.experimental.rf import RFModel as ModuleRFModel
from src.models.gbdt import GBDTModel as LegacyGBDTModel
from src.models.nn import NNModel as LegacyNNModel
from src.models.rf import RFModel as LegacyRFModel


def test_experimental_namespace_is_the_canonical_challenger_import_surface() -> None:
    assert GBDTModel is ModuleGBDTModel
    assert RFModel is ModuleRFModel
    assert NNModel is ModuleNNModel
    assert BayesGoalsModel is ModuleBayesGoalsModel


def test_legacy_top_level_challenger_imports_are_compatibility_shims() -> None:
    assert LegacyGBDTModel is GBDTModel
    assert LegacyRFModel is RFModel
    assert LegacyNNModel is NNModel
    assert LegacyBayesGoalsModel is BayesGoalsModel
    assert LegacyBayesBTModel is BayesStateSpaceBTModel
