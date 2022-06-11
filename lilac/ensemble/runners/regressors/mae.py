"""RMSE用のアンサンブル."""
from lilac.ensemble.ensemble_runner_base import EnsembleRunnerBase


class AveragingMaeEnsemble(EnsembleRunnerBase):
    """AveragingRegressorを使ってアンサンブルする."""

    def __init__(
        self, target_col, unused_cols, folds_gen_factory_settings, trainer_factory_settings, use_original_cols
    ):
        model_params = {"model_str": "avg_reg"}
        evaluator_flag = "mae"
        super().__init__(
            target_col,
            unused_cols,
            folds_gen_factory_settings,
            model_params,
            trainer_factory_settings,
            evaluator_flag,
            use_original_cols,
        )


class LgbmMaeEnsemble(EnsembleRunnerBase):
    """LgbmMaeを使ってアンサンブルする."""

    def __init__(
        self, target_col, unused_cols, folds_gen_factory_settings, trainer_factory_settings, use_original_cols
    ):
        model_params = {"model_str": "lgbm_mae"}
        evaluator_flag = "mae"
        super().__init__(
            target_col,
            unused_cols,
            folds_gen_factory_settings,
            model_params,
            trainer_factory_settings,
            evaluator_flag,
            use_original_cols,
        )
