from lilac.models import consts
from lilac.models.base.diff_ratio_base import DiffRegressorBase
from lilac.models.base.lgbm_base import (
    LgbmFairRegressorBase,
    LgbmMaeRegressorBase,
    LgbmRmseRegressorBase,
    LgbmRmsleRegressorBase,
)
from lilac.models.model_base import RegressorBase


class LgbmRegressor(RegressorBase):
    """LightGBMの回帰モデル."""

    def __init__(
        self,
        target_col,
        base_model,
        verbose_eval=consts.verbose_eval,
        early_stopping_rounds=consts.early_stopping_rounds,
        colsample_bytree=consts.colsample_bytree,
        reg_alpha=consts.reg_alpha,
        reg_lambda=consts.reg_lambda,
        subsample=consts.subsample,
        min_child_weight=consts.min_child_weight,
        n_estimators=consts.n_estimators,
        depth=consts.depth,
        seed=consts.seed,
        learning_rate=consts.learning_rate,
    ):
        super().__init__(target_col)
        lgbm_params = {
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "subsample": subsample,
            "min_child_weight": min_child_weight,
            "random_state": seed,
            "n_estimators": n_estimators,
            "max_depth": depth,
            "learning_rate": learning_rate,
        }
        self.model = base_model(verbose_eval, early_stopping_rounds, lgbm_params)

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        valid_x, valid_y = self.split_df2xy(valid_df)
        return self.model.fit(train_x, train_y, valid_x, valid_y)

    def _predict(self, test_df):
        """test_dfにtarget_colが入っていても大丈夫."""
        return self.model.predict(test_df)

    def return_flag(self):
        return self.model.return_flag()

    def get_importance(self):
        """lgbmのみ追加で実装している."""
        return self.model.get_importance()

    def get_additional(self):
        return {"importance": self.model.get_importance()}


class LgbmRmsleRegressor(RegressorBase):
    """目的関数がRMSLEのlgbm回帰モデル."""

    def __init__(
        self,
        target_col,
        verbose_eval=consts.verbose_eval,
        early_stopping_rounds=consts.early_stopping_rounds,
        colsample_bytree=consts.colsample_bytree,
        reg_alpha=consts.reg_alpha,
        reg_lambda=consts.reg_lambda,
        subsample=consts.subsample,
        min_child_weight=consts.min_child_weight,
        n_estimators=consts.n_estimators,
        depth=consts.depth,
        seed=consts.seed,
        learning_rate=consts.learning_rate,
    ):
        super().__init__(
            target_col=target_col,
            base_model=LgbmRmsleRegressorBase,
            verbose_eval=verbose_eval,
            early_stopping_rounds=early_stopping_rounds,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            subsample=subsample,
            min_child_weight=min_child_weight,
            n_estimators=n_estimators,
            depth=depth,
            seed=seed,
            learning_rate=learning_rate,
        )


class LgbmRmseRegressor(LgbmRegressor):
    """目的関数がRMSEのlgbm回帰モデル."""

    def __init__(
        self,
        target_col,
        verbose_eval=consts.verbose_eval,
        early_stopping_rounds=consts.early_stopping_rounds,
        colsample_bytree=consts.colsample_bytree,
        reg_alpha=consts.reg_alpha,
        reg_lambda=consts.reg_lambda,
        subsample=consts.subsample,
        min_child_weight=consts.min_child_weight,
        n_estimators=consts.n_estimators,
        depth=consts.depth,
        seed=consts.seed,
        learning_rate=consts.learning_rate,
    ):
        super().__init__(
            target_col=target_col,
            base_model=LgbmRmseRegressorBase,
            verbose_eval=verbose_eval,
            early_stopping_rounds=early_stopping_rounds,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            subsample=subsample,
            min_child_weight=min_child_weight,
            n_estimators=n_estimators,
            depth=depth,
            seed=seed,
            learning_rate=learning_rate,
        )


class LgbmMaeRegressor(LgbmRegressor):
    """目的関数がMAEのlgbm回帰モデル."""

    def __init__(
        self,
        target_col,
        verbose_eval=consts.verbose_eval,
        early_stopping_rounds=consts.early_stopping_rounds,
        colsample_bytree=consts.colsample_bytree,
        reg_alpha=consts.reg_alpha,
        reg_lambda=consts.reg_lambda,
        subsample=consts.subsample,
        min_child_weight=consts.min_child_weight,
        n_estimators=consts.n_estimators,
        depth=consts.depth,
        seed=consts.seed,
        learning_rate=consts.learning_rate,
    ):
        super().__init__(
            target_col=target_col,
            base_model=LgbmMaeRegressorBase,
            verbose_eval=verbose_eval,
            early_stopping_rounds=early_stopping_rounds,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            subsample=subsample,
            min_child_weight=min_child_weight,
            n_estimators=n_estimators,
            depth=depth,
            seed=seed,
            learning_rate=learning_rate,
        )


class LgbmFairRegressor(LgbmRegressor):
    """目的関数がfairのlgbm回帰モデル.MAE最適化の時に使われる.

    see:
    https://www.kaggle.com/c/allstate-claims-severity/discussion/24520
    """

    def __init__(
        self,
        target_col,
        verbose_eval=consts.verbose_eval,
        early_stopping_rounds=consts.early_stopping_rounds,
        colsample_bytree=consts.colsample_bytree,
        reg_alpha=consts.reg_alpha,
        reg_lambda=consts.reg_lambda,
        subsample=consts.subsample,
        min_child_weight=consts.min_child_weight,
        n_estimators=consts.n_estimators,
        depth=consts.depth,
        seed=consts.seed,
        learning_rate=consts.learning_rate,
    ):
        super().__init__(
            target_col=target_col,
            base_model=LgbmFairRegressorBase,
            verbose_eval=verbose_eval,
            early_stopping_rounds=early_stopping_rounds,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            subsample=subsample,
            min_child_weight=min_child_weight,
            n_estimators=n_estimators,
            depth=depth,
            seed=seed,
            learning_rate=learning_rate,
        )


class LgbmDiffMaeRegressor(DiffRegressorBase):
    """target_col-base_colに対してMAEで最適化するLGBMモデル."""

    def __init__(
        self,
        target_col,
        base_col,
        verbose_eval=consts.verbose_eval,
        early_stopping_rounds=consts.early_stopping_rounds,
        colsample_bytree=consts.colsample_bytree,
        reg_alpha=consts.reg_alpha,
        reg_lambda=consts.reg_lambda,
        subsample=consts.subsample,
        min_child_weight=consts.min_child_weight,
        n_estimators=consts.n_estimators,
        depth=consts.depth,
        seed=consts.seed,
        learning_rate=consts.learning_rate,
    ):
        model = LgbmMaeRegressor(
            target_col=target_col,
            verbose_eval=verbose_eval,
            early_stopping_rounds=early_stopping_rounds,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            subsample=subsample,
            min_child_weight=min_child_weight,
            n_estimators=n_estimators,
            depth=depth,
            seed=seed,
            learning_rate=learning_rate,
        )
        super().__init__(target_col=target_col, base_col=base_col, model=model)

    def get_importance(self):
        """lgbmのみ追加で実装している."""
        return self.model.get_importance()


class LgbmDiffRmseRegressor(DiffRegressorBase):
    """target_col-base_colに対してRMSEで最適化するLGBMモデル."""

    def __init__(
        self,
        target_col,
        base_col,
        verbose_eval=consts.verbose_eval,
        early_stopping_rounds=consts.early_stopping_rounds,
        colsample_bytree=consts.colsample_bytree,
        reg_alpha=consts.reg_alpha,
        reg_lambda=consts.reg_lambda,
        subsample=consts.subsample,
        min_child_weight=consts.min_child_weight,
        n_estimators=consts.n_estimators,
        depth=consts.depth,
        seed=consts.seed,
        learning_rate=consts.learning_rate,
    ):
        model = LgbmRmseRegressor(
            target_col=target_col,
            verbose_eval=verbose_eval,
            early_stopping_rounds=early_stopping_rounds,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            subsample=subsample,
            min_child_weight=min_child_weight,
            n_estimators=n_estimators,
            depth=depth,
            seed=seed,
            learning_rate=learning_rate,
        )
        super().__init__(target_col=target_col, base_col=base_col, model=model)

    def get_importance(self):
        """lgbmのみ追加で実装している."""
        return self.model.get_importance()
