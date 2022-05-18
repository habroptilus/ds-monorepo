from tools.ensemble.ensemble_runner_base import EnsembleRunnerBase


class LinearMultiEnsemble(EnsembleRunnerBase):
    """LrMultiClassifierを使ってアンサンブルする."""

    def __init__(self, target_col, unused_cols, folds_gen_settings, trainer_params, use_original_cols):
        model_params = {
            "model_str": "lr_multi",
            "target_col": target_col
        }
        evaluator_flag = "f1_macro"
        super().__init__(target_col, unused_cols, folds_gen_settings,
                         model_params, trainer_params, evaluator_flag, use_original_cols)


class AveragingMultiEnsemble(EnsembleRunnerBase):
    """AveragingMultiClassifierを使ってアンサンブルする."""

    def __init__(self, target_col, unused_cols, folds_gen_settings, trainer_params, use_original_cols):
        model_params = {
            "model_str": "avg_multi",
            "group_prefix": "pred",
            "target_col": target_col
        }
        evaluator_flag = "f1_macro"
        super().__init__(target_col, unused_cols, folds_gen_settings,
                         model_params, trainer_params, evaluator_flag, use_original_cols)


class LgbmMultiEnsemble(EnsembleRunnerBase):
    """多クラスlightgbmをつかってアンサンブルする.

    多クラス予測確率ベクトルを横に結合して次の層のモデルの特徴に使う.
    """

    def __init__(self, target_col, unused_cols, folds_gen_settings, trainer_params, use_original_cols):
        model_params = {
            "model_str": "lgbm_multi",
            "target_col": target_col
        }
        evaluator_flag = "f1_macro"
        super().__init__(target_col, unused_cols, folds_gen_settings,
                         model_params, trainer_params, evaluator_flag, use_original_cols)
