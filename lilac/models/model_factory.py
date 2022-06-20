from lilac.core.factory_base import FactoryBase

from .classifiers.averaging_classifiers import AveragingBinaryClassifier, AveragingMultiClassifier
from .classifiers.catb_classifiers import CatbBinaryClassifier, CatbMultiClassifier
from .classifiers.lgbm_classifiers import LgbmBinaryClassifier, LgbmMultiClassifier
from .classifiers.logistic_regression import LrMultiClassifier
from .regressors.averaging_regressors import AveragingRegressor
from .regressors.catb_regressors import CatbMaeRegressor, CatbRmseRegressor, CatbRmsleRegressor
from .regressors.lgbm_regressors import (
    LgbmDiffMaeRegressor,
    LgbmFairRegressor,
    LgbmMaeRegressor,
    LgbmRmseRegressor,
    LgbmRmsleRegressor,
)
from .regressors.linear_regressors import LinearModel, LinearPositiveModel, LinearRmsle, RidgeRmse, RidgeRmsle


class ModelFactory(FactoryBase):
    """予測モデル用ファクトリクラス."""

    def __init__(self, target_col, register_from=None):
        str2model = {
            "lgbm_rmsle": LgbmRmsleRegressor,
            "lgbm_rmse": LgbmRmseRegressor,
            "lgbm_mae": LgbmMaeRegressor,
            "lgbm_fair": LgbmFairRegressor,
            "lgbm_bin": LgbmBinaryClassifier,
            "lgbm_multi": LgbmMultiClassifier,
            "lgbm_diff_mae": LgbmDiffMaeRegressor,
            "catb_rmse": CatbRmseRegressor,
            "catb_rmsle": CatbRmsleRegressor,
            "catb_mae": CatbMaeRegressor,
            "catb_bin": CatbBinaryClassifier,
            "catb_multi": CatbMultiClassifier,
            "lr_rmsle": LinearRmsle,
            "lr_rmse": LinearModel,
            "lr_pos": LinearPositiveModel,
            "lr_multi": LrMultiClassifier,
            "ridge_rmse": RidgeRmse,
            "ridge_rmsle": RidgeRmsle,
            "avg_bin": AveragingBinaryClassifier,
            "avg_multi": AveragingMultiClassifier,
            "avg_reg": AveragingRegressor,
        }
        shared_params = {"target_col": target_col}
        super().__init__(str2model=str2model, register_from=register_from, shared_params=shared_params)
