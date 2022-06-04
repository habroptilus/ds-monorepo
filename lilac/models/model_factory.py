from lilac.core.factory_base import FactoryBase

from .classifiers.averaging_classifiers import AveragingBinaryClassifier, AveragingMultiClassifier
from .classifiers.catb_classifiers import CatbBinaryClassifier, CatbMultiClassifier
from .classifiers.lgbm_classifiers import LgbmBinaryClassifier, LgbmMultiClassifier
from .classifiers.logistic_regression import LrMultiClassifier
from .regressors.averaging_regressors import AveragingRegressor
from .regressors.catb_regressors import CatbRmseRegressor, CatbRmsleRegressor
from .regressors.lgbm_regressors import LgbmMaeRegressor, LgbmRmseRegressor, LgbmRmsleRegressor
from .regressors.linear_regressors import LinearModel, LinearPositiveModel, LinearRmsle, RidgeRmse, RidgeRmsle


class ModelFactory(FactoryBase):
    """予測モデル用ファクトリクラス."""

    def __init__(self, target_col, register_from=None):
        str2model = {
            "lgbm_rmsle": LgbmRmsleRegressor,
            "lgbm_rmse": LgbmRmseRegressor,
            "lgbm_mae": LgbmMaeRegressor,
            "lgbm_bin": LgbmBinaryClassifier,
            "lgbm_multi": LgbmMultiClassifier,
            "catb_rmse": CatbRmseRegressor,
            "catb_rmsle": CatbRmsleRegressor,
            "catb_bin": CatbBinaryClassifier,
            "catb_multi": CatbMultiClassifier,
            "linear_rmsle": LinearRmsle,
            "linear_rmse": LinearModel,
            "linear_pos": LinearPositiveModel,
            "ridge_rmse": RidgeRmse,
            "ridge_rmsle": RidgeRmsle,
            "lr_multi": LrMultiClassifier,
            "avg_bin": AveragingBinaryClassifier,
            "avg_multi": AveragingMultiClassifier,
            "avg_reg": AveragingRegressor,
        }
        shared_params = {"target_col": target_col}
        super().__init__(str2model=str2model, register_from=register_from, shared_params=shared_params)
