from src.models.bayes_state_space_goals import BayesGoalsModel as LegacyBayesGoalsModel
from src.models.bayes_state_space_bt import BayesStateSpaceBTModel as LegacyBayesBTModel
from src.models.challenger_prob import GAMSplineModel as LegacyGAMSplineModel
from src.models.challenger_prob import MARSHingeModel as LegacyMARSHingeModel
from src.models.challenger_prob import VanillaGLMModel as LegacyVanillaGLMModel
from src.models.core import (
    GLMElasticNetModel,
    GLMLassoModel,
    GLMRidgeModel,
    LassoCredibilityModel,
    VanillaGLMModel,
    build_penalized_glm,
)
from src.models.core.glm_penalized import build_penalized_glm as ModuleBuildPenalizedGLM
from src.models.glm_elastic_net import GLMElasticNetModel as LegacyGLMElasticNetModel
from src.models.glm_lasso import GLMLassoModel as LegacyGLMLassoModel
from src.models.glm_penalized import GLMRidgeModel as LegacyPenalizedGLMRidgeModel
from src.models.glm_penalized import build_penalized_glm as LegacyBuildPenalizedGLM
from src.models.glm_ridge import GLMRidgeModel as LegacyGLMRidgeModel
from src.models.lasso_credibility import LassoCredibilityModel as LegacyLassoCredibilityModel
from src.models.core.glm_penalized import GLMLassoModel as ModuleGLMLassoModel
from src.models.core.glm_penalized import GLMRidgeModel as ModuleGLMRidgeModel
from src.models.core.glm_vanilla import VanillaGLMModel as ModuleVanillaGLMModel
from src.models.core.lasso_credibility import LassoCredibilityModel as ModuleLassoCredibilityModel
from src.models.ensemble_stack import StackingEnsemble as LegacyStackingEnsemble
from src.models.experimental import BayesGoalsModel, BayesStateSpaceBTModel, GBDTModel, MARSHingeModel, NNModel, RFModel
from src.models.experimental.bayes_state_space_goals import BayesGoalsModel as ModuleBayesGoalsModel
from src.models.experimental.gbdt import GBDTModel as ModuleGBDTModel
from src.models.experimental.mars_hinge import MARSHingeModel as ModuleMARSHingeModel
from src.models.experimental.nn import NNModel as ModuleNNModel
from src.models.experimental.rf import RFModel as ModuleRFModel
from src.models.extensions.ensemble_stack import StackingEnsemble
from src.models.extensions.glm_variants import GAMSplineModel
from src.models.gbdt import GBDTModel as LegacyGBDTModel
from src.models.nn import NNModel as LegacyNNModel
from src.models.rf import RFModel as LegacyRFModel


def test_experimental_namespace_is_the_canonical_challenger_import_surface() -> None:
    assert GBDTModel is ModuleGBDTModel
    assert RFModel is ModuleRFModel
    assert NNModel is ModuleNNModel
    assert BayesGoalsModel is ModuleBayesGoalsModel
    assert MARSHingeModel is ModuleMARSHingeModel
    assert GAMSplineModel is not MARSHingeModel


def test_core_namespace_is_the_canonical_cas_import_surface() -> None:
    assert GLMRidgeModel is ModuleGLMRidgeModel
    assert GLMLassoModel is ModuleGLMLassoModel
    assert LassoCredibilityModel is ModuleLassoCredibilityModel
    assert VanillaGLMModel is ModuleVanillaGLMModel
    assert build_penalized_glm is ModuleBuildPenalizedGLM


def test_legacy_top_level_core_imports_are_compatibility_shims() -> None:
    assert LegacyGLMRidgeModel is GLMRidgeModel
    assert LegacyPenalizedGLMRidgeModel is GLMRidgeModel
    assert LegacyGLMLassoModel is GLMLassoModel
    assert LegacyGLMElasticNetModel is GLMElasticNetModel
    assert LegacyLassoCredibilityModel is LassoCredibilityModel
    assert LegacyBuildPenalizedGLM is build_penalized_glm


def test_legacy_top_level_challenger_imports_are_compatibility_shims() -> None:
    assert LegacyGBDTModel is GBDTModel
    assert LegacyRFModel is RFModel
    assert LegacyNNModel is NNModel
    assert LegacyBayesGoalsModel is BayesGoalsModel
    assert LegacyBayesBTModel is BayesStateSpaceBTModel
    assert LegacyMARSHingeModel is MARSHingeModel
    assert LegacyGAMSplineModel is GAMSplineModel
    assert LegacyVanillaGLMModel is VanillaGLMModel
    assert LegacyStackingEnsemble is StackingEnsemble
