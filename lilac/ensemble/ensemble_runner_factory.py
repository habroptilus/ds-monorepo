from lilac.core.factory_base import FactoryBase
from lilac.ensemble.runners.classifiers.multi import AveragingMultiEnsemble, LgbmMultiEnsemble, LinearMultiEnsemble
from lilac.ensemble.runners.regressors.mae import AveragingMaeEnsemble, LgbmMaeEnsemble
from lilac.ensemble.runners.regressors.rmse import (
    AveragingRmseEnsemble,
    LgbmRmseEnsemble,
    LinearRmseEnsemble,
    RidgeRmseEnsemble,
)
from lilac.ensemble.runners.regressors.rmsle import (
    AveragingRmsleEnsemble,
    LgbmRmsleEnsemble,
    LinearRmsleEnsemble,
    RidgeRmsleEnsemble,
)


class EnsembleRunnerFactory(FactoryBase):
    """アンサンブル実行クラスのFactory."""

    def __init__(self, register_from=None):
        str2model = {
            "lr_multi": LinearMultiEnsemble,
            "lr_rmse": LinearRmseEnsemble,
            "lr_rmsle": LinearRmsleEnsemble,
            "lgbm_rmsle": LgbmRmsleEnsemble,
            "lgbm_rmse": LgbmRmseEnsemble,
            "lgbm_mae": LgbmMaeEnsemble,
            "lgbm_multi": LgbmMultiEnsemble,
            "ridge_rmsle": RidgeRmsleEnsemble,
            "ridge_rmse": RidgeRmseEnsemble,
            "avg_rmsle": AveragingRmsleEnsemble,
            "avg_rmse": AveragingRmseEnsemble,
            "avg_mae": AveragingMaeEnsemble,
            "avg_multi": AveragingMultiEnsemble,
        }
        super().__init__(str2model, register_from)
