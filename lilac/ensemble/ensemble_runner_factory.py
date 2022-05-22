from lilac.ensemble.runners.classifiers.multi import LgbmMultiEnsemble, LinearMultiEnsemble, AveragingMultiEnsemble
from lilac.ensemble.runners.regressors.rmse import LgbmRmseEnsemble, LinearRmseEnsemble, RidgeRmseEnsemble, AveragingRmseEnsemble
from lilac.ensemble.runners.regressors.rmsle import LgbmRmsleEnsemble, LinearRmsleEnsemble, RidgeRmsleEnsemble, AveragingRmsleEnsemble
from lilac.core.factory_base import FactoryBase


class EnsembleRunnerFactory(FactoryBase):
    """アンサンブル実行クラスのFactory."""

    def __init__(self, register_from=None):
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
        super().__init__(str2model, register_from)
