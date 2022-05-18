from .regressors.lgbm_regressors import LgbmRmsleRegressor, LgbmRmseRegressor, LgbmMaeRegressor
from .regressors.linear_regressors import LinearRmsle, LinearModel, LinearPositiveModel, RidgeRmsle, RidgeRmse
from .regressors.catb_regressors import CatbRmseRegressor, CatbRmsleRegressor
from .regressors.averaging_regressors import AveragingRegressor
from .classifiers.lgbm_classifiers import LgbmBinaryClassifier, LgbmMultiClassifier
from .classifiers.logistic_regression import LrMultiClassifier
from .classifiers.catb_classifiers import CatbBinaryClassifier, CatbMultiClassifier
from .classifiers.averaging_classifiers import AveragingBinaryClassifier, AveragingMultiClassifier
from tools.core.factory_base import FactoryBase


class ModelFactory(FactoryBase):
    def __init__(self, custom_members=None):
        # これfactoryからmodel側に移す
        self.default_params = {
            "verbose_eval": 100,
            "early_stopping_rounds": 100,
            "class_weight": "balanced",
            "lgbm_params": {
                "colsample_bytree": 0.8,
                "reg_alpha": 0,
                "reg_lambda": 0,
                "subsample": 0.8,
                "min_child_weight": 1.0,
                "num_leaves": int(2 ** 5 * 0.7)
            },
            "catb_params": {
                'learning_rate': 0.1,
                'random_strength': 1,
                'bagging_temperature': 0.1,
                'od_type': "IncToDec",
                'od_wait': 10
            }
        }
        str2model = {
            "lgbm_rmsle": LgbmRmsleRegressor,
            "lgbm_rmse": LgbmRmseRegressor,
            "lgbm_mae": LgbmMaeRegressor,
            "lgbm_bin":  LgbmBinaryClassifier,
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
            "avg_reg": AveragingRegressor
        }
        super().__init__(str2model, custom_members)

    def run(self, model_str, target_col, params=None, depth=5, n_estimators=2000, seed=None):
        """todo: これもmodelのデフォルトパラメータをここに置くのはおかしい"""
        return super().run(model_str, target_col, params, depth, n_estimators, seed)

    def get_params(self, target_col, params, depth, n_estimators, seed):
        result = self.default_params.copy()
        if params:
            print(f"Params updated with {params}")
            self.params.update(params)
        result["target_col"] = target_col
        result["seed"] = seed
        result["lgbm_params"]["random_state"] = seed
        result["catb_params"]["random_seed"] = seed
        result["lgbm_params"]["n_estimators"] = n_estimators
        result["catb_params"]["num_boost_round"] = n_estimators
        result["lgbm_params"]["max_depth"] = depth
        result["catb_params"]["depth"] = depth
        return result
