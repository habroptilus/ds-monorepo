from tools.ensemble.ensemble_runner_base import EnsembleRunnerBase


class LinearMultiEnsemble(EnsembleRunnerBase):
    """LrMultiClassifierを使ってアンサンブルする."""

    def __init__(self, target_col, unused_cols, folds_gen_factory_settings, trainer_factory_settings, use_original_cols):
        model_params = {
            "model_str": "lr_multi"
        }
        evaluator_flag = "f1_macro"
        super().__init__(target_col, unused_cols, folds_gen_factory_settings,
                         model_params, trainer_factory_settings, evaluator_flag, use_original_cols)


class AveragingMultiEnsemble(EnsembleRunnerBase):
    """AveragingMultiClassifierを使ってアンサンブルする."""

    def __init__(self, target_col, unused_cols, folds_gen_factory_settings, trainer_factory_settings, use_original_cols):
        model_params = {
            "model_str": "avg_multi",
            "params":
            {
                "group_prefix": "pred"
            }
        }
        evaluator_flag = "f1_macro"
        super().__init__(target_col, unused_cols, folds_gen_factory_settings,
                         model_params, trainer_factory_settings, evaluator_flag, use_original_cols)


class LgbmMultiEnsemble(EnsembleRunnerBase):
    """多クラスlightgbmをつかってアンサンブルする.

    多クラス予測確率ベクトルを横に結合して次の層のモデルの特徴に使う.
    """

    def __init__(self, target_col, unused_cols, folds_gen_factory_settings, trainer_factory_settings, use_original_cols):
        model_params = {
            "model_str": "lgbm_multi"
        }
        evaluator_flag = "f1_macro"
        super().__init__(target_col, unused_cols, folds_gen_factory_settings,
                         model_params, trainer_factory_settings, evaluator_flag, use_original_cols)
