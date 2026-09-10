"""Module containing all drug response prediction models."""

__all__ = [
    "MULTI_DRUG_MODEL_FACTORY",
    "SINGLE_DRUG_MODEL_FACTORY",
    "MODEL_FACTORY",
    "NaivePredictor",
    "NaiveDrugMeanPredictor",
    "NaiveCellLineMeanPredictor",
    "NaiveTissueMeanPredictor",
    "NaiveTissueDrugMeanPredictor",
    "NaiveMeanEffectsPredictor",
    "ElasticNetModel",
    "RandomForest",
    "RandomForestTargetMut",
    "ElasticNetTargetMut",
    "RandomForestTargetMut893",
    "ElasticNetTargetMut893",
    "SingleDrugRandomForestTargetMut",
    "RandomForest893",
    "RandomForestAE",
    "RandomForestAEMut",
    "RandomForestMolGNet",
    "RandomForestMolGNetMut",
    "RandomForestAEMolGNet",
    "RandomForestAEMolGNetMut",
    "RandomForestAEMolGNetMutCW",
    "RandomForestAEMolGNetMutTrim",
    "RandomForestAEMolGNetMutBJ",
    "RandomForestAEMolGNetMutDrugTrim",
    "RandomForestAEMolGNetMutKern8",
    "ElasticNetGeneExpr",
    "ElasticNetGeneExpr893",
    "ElasticNetResid",
    "ElasticNetResid893",
    "ElasticNetTargetMutResid",
    "ElasticNetTargetMutResid893",
    "SVMRegressor",
    "SimpleNeuralNetwork",
    "MultiViewNeuralNetwork",
    "MultiViewRandomForest",
    "SingleDrugRandomForest",
    "SingleDrugElasticNet",
    "SRMF",
    "EnsembleMF",
    "GradientBoosting",
    "MOLIR",
    "SuperFELTR",
    "DIPKModel",
    "DIPK11883Model",
    "DIPKLOGModel",
    "DIPKTCGAModel",
    "DIPKLOG2270Model",
    "DrugGNN",
    "PharmaFormerModel",
    "PrecilyModel",
    "KNNRegressor",
    "AdaBoostDecisionTree",
    "LassoModel",
    "MultiViewXGBoost",
    "MultiViewLightGBM",
    "SparseGO",
]

from .baselines.multi_view_lightgbm import MultiViewLightGBM
from .baselines.buckley_james import RandomForestAEMolGNetMutBJ
from .baselines.curve_quality_weights import (
    RandomForestAEMolGNetMutCW,
    RandomForestAEMolGNetMutTrim,
)
from .baselines.dipk_feature_sklearn import (
    RandomForest893,
    RandomForestAE,
    RandomForestAEMolGNet,
    RandomForestAEMolGNetMut,
    RandomForestAEMolGNetMutKern8,
    RandomForestAEMut,
    RandomForestMolGNet,
    RandomForestMolGNetMut,
)
from .baselines.drug_level_trim import RandomForestAEMolGNetMutDrugTrim
from .baselines.multi_view_random_forest import MultiViewRandomForest
from .baselines.multi_view_xgboost import MultiViewXGBoost
from .baselines.naive_pred import (
    NaiveCellLineMeanPredictor,
    NaiveDrugMeanPredictor,
    NaiveMeanEffectsPredictor,
    NaivePredictor,
    NaiveTissueDrugMeanPredictor,
    NaiveTissueMeanPredictor,
)
from .baselines.singledrug_baselines import SingleDrugElasticNet, SingleDrugRandomForest
from .baselines.sklearn_models import (
    AdaBoostDecisionTree,
    ElasticNetModel,
    GradientBoosting,
    KNNRegressor,
    LassoModel,
    RandomForest,
    SVMRegressor,
)
from .baselines.drug_resid_sklearn import (
    ElasticNetGeneExpr,
    ElasticNetGeneExpr893,
    ElasticNetResid,
    ElasticNetResid893,
    ElasticNetTargetMutResid,
    ElasticNetTargetMutResid893,
)
from .baselines.target_mut_sklearn import (
    ElasticNetTargetMut,
    ElasticNetTargetMut893,
    RandomForestTargetMut,
    RandomForestTargetMut893,
    SingleDrugRandomForestTargetMut,
)
from .DIPK.dipk import DIPKModel
from .DIPK11883.dipk11883 import DIPK11883Model
from .DIPKLOG.dipklog import DIPKLOGModel
from .DIPKLOG2270.dipklog2270 import DIPKLOG2270Model
from .DIPKTCGA.dipktcga import DIPKTCGAModel
from .drp_model import DRPModel
from .DrugGNN import DrugGNN
from .EnsembleMF import EnsembleMF
from .MOLIR.molir import MOLIR
from .PharmaFormer.pharmaformer import PharmaFormerModel
from .Precily import PrecilyModel
from .SimpleNeuralNetwork.multi_view_neural_network import MultiViewNeuralNetwork
from .SimpleNeuralNetwork.simple_neural_network import SimpleNeuralNetwork
from .SparseGO.sparsego import SparseGOModel
from .SRMF.srmf import SRMF
from .SuperFELTR.superfeltr import SuperFELTR

# SINGLE_DRUG_MODEL_FACTORY is used in the pipeline!
SINGLE_DRUG_MODEL_FACTORY: dict[str, type[DRPModel]] = {
    "SingleDrugElasticNet": SingleDrugElasticNet,
    "SingleDrugRandomForest": SingleDrugRandomForest,
    "SingleDrugRandomForestTargetMut": SingleDrugRandomForestTargetMut,
    "MOLIR": MOLIR,
    "SuperFELTR": SuperFELTR,
}

# MULTI_DRUG_MODEL_FACTORY is used in the pipeline!
MULTI_DRUG_MODEL_FACTORY: dict[str, type[DRPModel]] = {
    # Naive predictors
    "NaivePredictor": NaivePredictor,
    "NaiveCellLineMeanPredictor": NaiveCellLineMeanPredictor,
    "NaiveDrugMeanPredictor": NaiveDrugMeanPredictor,
    "NaiveMeanEffectsPredictor": NaiveMeanEffectsPredictor,
    "NaiveTissueMeanPredictor": NaiveTissueMeanPredictor,
    "NaiveTissueDrugMeanPredictor": NaiveTissueDrugMeanPredictor,
    # Sklearn Baselines
    "AdaBoostDecisionTree": AdaBoostDecisionTree,
    "ElasticNet": ElasticNetModel,
    "Lasso": LassoModel,
    "GradientBoosting": GradientBoosting,
    "KNNRegressor": KNNRegressor,
    "RandomForest": RandomForest,
    "RandomForestTargetMut": RandomForestTargetMut,
    "ElasticNetTargetMut": ElasticNetTargetMut,
    "RandomForestTargetMut893": RandomForestTargetMut893,
    "ElasticNetTargetMut893": ElasticNetTargetMut893,
    "ElasticNetGeneExpr": ElasticNetGeneExpr,
    "ElasticNetGeneExpr893": ElasticNetGeneExpr893,
    "ElasticNetResid": ElasticNetResid,
    "ElasticNetResid893": ElasticNetResid893,
    "ElasticNetTargetMutResid": ElasticNetTargetMutResid,
    "ElasticNetTargetMutResid893": ElasticNetTargetMutResid893,
    "MultiViewRandomForest": MultiViewRandomForest,
    # RF auf DIPKs Feature-Bausteinen (AE + gepooltes MolGNet), Ablations-Faktorial 2026-08-06
    "RandomForest893": RandomForest893,
    "RandomForestAE": RandomForestAE,
    "RandomForestAEMut": RandomForestAEMut,
    "RandomForestMolGNet": RandomForestMolGNet,
    "RandomForestMolGNetMut": RandomForestMolGNetMut,
    "RandomForestAEMolGNet": RandomForestAEMolGNet,
    "RandomForestAEMolGNetMut": RandomForestAEMolGNetMut,
    "RandomForestAEMolGNetMutCW": RandomForestAEMolGNetMutCW,
    "RandomForestAEMolGNetMutTrim": RandomForestAEMolGNetMutTrim,
    "RandomForestAEMolGNetMutBJ": RandomForestAEMolGNetMutBJ,
    "RandomForestAEMolGNetMutDrugTrim": RandomForestAEMolGNetMutDrugTrim,
    "RandomForestAEMolGNetMutKern8": RandomForestAEMolGNetMutKern8,
    "SVR": SVMRegressor,
    # Other Baselines
    "DrugGNN": DrugGNN,
    "SimpleNeuralNetwork": SimpleNeuralNetwork,
    "MultiViewNeuralNetwork": MultiViewNeuralNetwork,
    "MultiViewXGBoost": MultiViewXGBoost,
    "MultiViewLightGBM": MultiViewLightGBM,
    # Published models
    "DIPK": DIPKModel,
    "DIPK11883": DIPK11883Model,  # DIPK auf dem TCGA-Autoencoder-Genraum (11.883 statt 2.270 Gene)
    "DIPKLOG": DIPKLOGModel,  # dito, aber log2(TPM+1) statt linearer TPM (Kontrolle Aufgabe B)
    "DIPKTCGA": DIPKTCGAModel,  # dito, Gen-Autoencoder auf TCGA vortrainiert (Aufgabe B Stufe 2)
    "DIPKLOG2270": DIPKLOG2270Model,  # Stock-Genliste (2.270), aber log2(TPM+1) -- isoliert den Skaleneffekt
    "PharmaFormer": PharmaFormerModel,
    "SRMF": SRMF,
    "EnsembleMF": EnsembleMF,
    "Precily": PrecilyModel,
    "SparseGO": SparseGOModel,
}

# MODEL_FACTORY is used in the pipeline!
MODEL_FACTORY = MULTI_DRUG_MODEL_FACTORY.copy()
MODEL_FACTORY.update(SINGLE_DRUG_MODEL_FACTORY)
