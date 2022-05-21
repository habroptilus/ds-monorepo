"""RMSE用のアンサンブル."""
from tools.ensemble.ensemble_runner_base import EnsembleRunnerBase


class RidgeRmseEnsemble(EnsembleRunnerBase):
    """RidgeRmseを使ってアンサンブルする."""

    def __init__(self, target_col, unused_cols, folds_gen_factory_settings,
                 trainer_factory_settings, use_original_cols):
        model_params = {
            "model_str": "ridge_rmse",
        }
        evaluator_flag = "rmse"
        super().__init__(target_col, unused_cols, folds_gen_factory_settings,
                         model_params, trainer_factory_settings, evaluator_flag, use_original_cols)


class LgbmRmseEnsemble(EnsembleRunnerBase):
    """LgbmRmseを使ってアンサンブルする."""

    def __init__(self, target_col, unused_cols, folds_gen_factory_settings, trainer_factory_settings, use_original_cols):
        model_params = {
            "model_str": "lgbm_rmse"
        }
        evaluator_flag = "rmse"
        super().__init__(target_col, unused_cols, folds_gen_factory_settings,
                         model_params, trainer_factory_settings, evaluator_flag, use_original_cols)


class LinearRmseEnsemble(EnsembleRunnerBase):
    """LinearRmseを使ってアンサンブルする."""

    def __init__(self, target_col, unused_cols, folds_gen_factory_settings, trainer_factory_settings, use_original_cols):
        model_params = {
            "model_str": "linear_rmse"
        }
        evaluator_flag = "rmse"
        super().__init__(target_col, unused_cols, folds_gen_factory_settings,
                         model_params, trainer_factory_settings, evaluator_flag, use_original_cols)


class AveragingRmseEnsemble(EnsembleRunnerBase):
    """AveragingRegressorを使ってアンサンブルする."""

    def __init__(self, target_col, unused_cols, folds_gen_factory_settings, trainer_factory_settings, use_original_cols):
        model_params = {
            "model_str": "avg_reg"
        }
        evaluator_flag = "rmse"
        super().__init__(target_col, unused_cols, folds_gen_factory_settings,
                         model_params, trainer_factory_settings, evaluator_flag, use_original_cols)
