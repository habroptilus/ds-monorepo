from lilac.core.factory_base import FactoryBase
from lilac.evaluators.evaluator_base import (
    AccuracyEvaluator,
    AucEvaluator,
    MacroF1Evaluator,
    MaeEvaluator,
    MapeEvaluator,
    PrAucEvaluator,
    RmseEvaluator,
    RmsleEvaluator,
    RmspeEvaluator,
)


class EvaluatorFactory(FactoryBase):
    def __init__(self, register_from=None):
        str2model = {
            "rmsle": RmsleEvaluator,
            "rmse": RmseEvaluator,
            "mae": MaeEvaluator,
            "auc": AucEvaluator,
            "accuracy": AccuracyEvaluator,
            "f1_macro": MacroF1Evaluator,
            "prauc": PrAucEvaluator,
            "mape": MapeEvaluator,
            "rmspe": RmspeEvaluator,
        }
        super().__init__(str2model, register_from)
