from tools.evaluators.evaluator_base import RmsleEvaluator, RmseEvaluator, MaeEvaluator, AucEvaluator, AccuracyEvaluator, MacroF1Evaluator
from tools.core.factory_base import FactoryBase


class EvaluatorFactory(FactoryBase):
    def __init__(self, register_from=None):
        str2model = {
            "rmsle": RmsleEvaluator,
            "rmse": RmseEvaluator,
            "mae": MaeEvaluator,
            "auc": AucEvaluator,
            "accuracy": AccuracyEvaluator,
            "f1_macro": MacroF1Evaluator
        }
        super().__init__(str2model, register_from)
