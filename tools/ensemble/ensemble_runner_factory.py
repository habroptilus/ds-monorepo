from tools.ensemble.runners.classifiers.multi import LgbmMultiEnsemble, LinearMultiEnsemble, AveragingMultiEnsemble
from tools.ensemble.runners.regressors.rmse import LgbmRmseEnsemble, LinearRmseEnsemble, RidgeRmseEnsemble, AveragingRmseEnsemble
from tools.ensemble.runners.regressors.rmsle import LgbmRmsleEnsemble, LinearRmsleEnsemble, RidgeRmsleEnsemble, AveragingRmsleEnsemble
from tools.core.factory_base import FactoryBase


class EnsembleRunnerFactory(FactoryBase):
    def __init__(self):
        str2model = {
            "lgbm_rmsle": LgbmRmsleEnsemble,
            "linear_rmsle": LinearRmsleEnsemble,
            "ridge_rmsle": RidgeRmsleEnsemble,
            "avg_rmsle": AveragingRmsleEnsemble,
            "lgbm_rmse": LgbmRmseEnsemble,
            "linear_rmse": LinearRmseEnsemble,
            "ridge_rmse": RidgeRmseEnsemble,
            "avg_rmse": AveragingRmseEnsemble,
            "lgbm_multi": LgbmMultiEnsemble,
            "lr_multi": LinearMultiEnsemble,
            "avg_multi": AveragingMultiEnsemble
        }
        super().__init__(str2model)

    def get_params(self, params=None):
        return params or {}

    def run(self, model_str, params):
        return super().run(model_str, params)
