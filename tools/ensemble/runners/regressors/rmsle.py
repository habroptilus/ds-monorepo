from tools.ensemble.ensemble_runner_base import EnsembleRunnerBase, LrRmsleEnsembleRunnerBase


class AveragingRmsleEnsemble(EnsembleRunnerBase):
    """AveragingRegressorを使ってアンサンブルする."""

    def __init__(self, target_col, unused_cols, folds_gen_settings, trainer_params, use_original_cols):
        model_params = {
            "model_str": "avg_reg",
            "target_col": target_col
        }
        evaluator_flag = "rmsle"
        super().__init__(target_col, unused_cols, folds_gen_settings,
                         model_params, trainer_params, evaluator_flag, use_original_cols)


class LinearRmsleEnsemble(LrRmsleEnsembleRunnerBase):
    """LinearRmsleを使ってアンサンブルする.予測値はlogをとって入力とする."""

    def __init__(self, target_col, unused_cols, folds_gen_settings, trainer_params, use_original_cols):
        model_params = {
            "model_str": "linear_rmsle",
            "target_col": target_col
        }
        evaluator_flag = "rmsle"
        super().__init__(target_col, unused_cols, folds_gen_settings,
                         model_params, trainer_params, evaluator_flag, use_original_cols)


class RidgeRmsleEnsemble(LrRmsleEnsembleRunnerBase):
    """RidgeRmsleを使ってアンサンブルする予測値はlogをとって入力とする."""

    def __init__(self, target_col, unused_cols, folds_gen_settings, trainer_params, use_original_cols):
        model_params = {
            "model_str": "ridge_rmsle",
            "target_col": target_col
        }
        evaluator_flag = "rmsle"
        super().__init__(target_col, unused_cols, folds_gen_settings,
                         model_params, trainer_params, evaluator_flag, use_original_cols)


class LgbmRmsleEnsemble(EnsembleRunnerBase):
    """LgbmRmsleを使ってアンサンブルする."""

    def __init__(self, target_col, unused_cols, folds_gen_settings, trainer_params, use_original_cols):
        model_params = {
            # todo: この辺ちゃんとパラメータ与えた方が良いかも？デフォルトパラメータの持ちかたにもよるので後で考える.
            "model_str": "lgbm_rmsle",
            "target_col": target_col
        }
        evaluator_flag = "rmsle"
        super().__init__(target_col, unused_cols, folds_gen_settings,
                         model_params, trainer_params, evaluator_flag, use_original_cols)
