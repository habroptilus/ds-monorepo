from lilac.models import consts
from lilac.models.base.lgbm_base import LgbmBinaryClassifierBase, LgbmMultiClassifierBase, LgbmXentropyClassifierBase
from lilac.models.model_base import BinaryClassifierBase, MultiClassifierBase


class LgbmBinaryClassifier(BinaryClassifierBase):
    """目的関数がLoglossのlgbm2値分類モデル."""

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
        min_child_samples=consts.min_child_samples,
        class_weight=consts.class_weight,
    ):
        super().__init__(target_col)
        lgbm_params = {
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "subsample": subsample,
            "min_child_weight": min_child_weight,
            "random_state": seed,
            "max_depth": depth,
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
        }

        self.model = LgbmBinaryClassifierBase(
            verbose_eval, early_stopping_rounds, n_estimators, lgbm_params, class_weight
        )

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        valid_x, valid_y = self.split_df2xy(valid_df)
        return self.model.fit(train_x, train_y, valid_x, valid_y)

    def _predict_proba(self, test_df):
        """test_dfにtarget_colが入っていても大丈夫."""
        return self.model.predict_proba(test_df)

    def return_flag(self):
        return f"{self.model.return_flag()}_bin"

    def get_importance(self):
        """lgbmのみ追加で実装している."""
        return self.model.get_importance()

    def get_additional(self):
        return {"importance": self.model.get_importance()}


class LgbmMultiClassifier(MultiClassifierBase):
    """目的関数がLoglossのlgbm多値分類モデル."""

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
        min_child_samples=consts.min_child_samples,
        class_weight=consts.class_weight,
    ):
        super().__init__(target_col)
        lgbm_params = {
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "subsample": subsample,
            "min_child_weight": min_child_weight,
            "random_state": seed,
            "max_depth": depth,
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
        }
        self.model = LgbmMultiClassifierBase(
            verbose_eval, early_stopping_rounds, n_estimators, lgbm_params, class_weight
        )

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        valid_x, valid_y = self.split_df2xy(valid_df)
        return self.model.fit(train_x, train_y, valid_x, valid_y)

    def _predict_proba(self, test_df):
        """test_dfにtarget_colが入っていても大丈夫."""
        return self.model.predict_proba(test_df)

    def return_flag(self):
        return f"{self.model.return_flag()}_multi"

    def get_importance(self):
        """lgbmのみ追加で実装している."""
        return self.model.get_importance()

    def get_additional(self):
        return {"importance": self.model.get_importance()}


class LgbmXentropyClassifier(BinaryClassifierBase):
    """目的関数がXentropyのlgbm2値分類モデル."""

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
        min_child_samples=consts.min_child_samples,
        class_weight=consts.class_weight,
    ):
        super().__init__(target_col)
        lgbm_params = {
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "subsample": subsample,
            "min_child_weight": min_child_weight,
            "random_state": seed,
            "max_depth": depth,
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
        }

        self.model = LgbmXentropyClassifierBase(
            verbose_eval, early_stopping_rounds, n_estimators, lgbm_params, class_weight
        )

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        valid_x, valid_y = self.split_df2xy(valid_df)
        return self.model.fit(train_x, train_y, valid_x, valid_y)

    def _predict_proba(self, test_df):
        """test_dfにtarget_colが入っていても大丈夫."""
        return self.model.predict_proba(test_df)

    def return_flag(self):
        return f"{self.model.return_flag()}_bin"

    def get_importance(self):
        """lgbmのみ追加で実装している."""
        return self.model.get_importance()

    def get_additional(self):
        return {"importance": self.model.get_importance()}
