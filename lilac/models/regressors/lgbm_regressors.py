from lilac.models.model_base import RegressorBase
from lilac.models.base.lgbm_base import _LgbmRegressor, _LgbmRmsleRegressor


class LgbmRmsleRegressor(RegressorBase):
    """目的関数がRMSLEのlgbm回帰モデル."""

    def __init__(self, target_col, verbose_eval=100, early_stopping_rounds=100, colsample_bytree=0.8,
                 reg_alpha=0, reg_lambda=0, subsample=0.8, min_child_weight=1.0, num_leaves=int(2 ** 5 * 0.7),
                 n_estimators=2000, depth=5, seed=None):
        super().__init__(target_col)
        lgbm_params = {
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "subsample": subsample,
            "min_child_weight": min_child_weight,
            "num_leaves": num_leaves,
            "random_state": seed,
            "n_estimators": n_estimators,
            "max_depth": depth
        }

        self.model = _LgbmRmsleRegressor(
            verbose_eval, early_stopping_rounds, lgbm_params)

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        valid_x, valid_y = self.split_df2xy(valid_df)

        return self.model.fit(train_x, train_y, valid_x, valid_y)

    def _predict(self, test_df):
        """test_dfにtarget_colが入っていても大丈夫."""
        return self.model.predict(test_df)

    def get_importance(self):
        """lgbmのみ追加で実装している."""
        return self.model.get_importance()

    def return_flag(self):
        """RMSLEはそのままだとRMSEと同じになってしまうのでつける"""
        return f"{self.model.return_flag()}_rmsle"


class LgbmRmseRegressor(RegressorBase):
    """目的関数がRMSEのlgbm回帰モデル."""

    def __init__(self, target_col, verbose_eval=100, early_stopping_rounds=100, colsample_bytree=0.8,
                 reg_alpha=0, reg_lambda=0, subsample=0.8, min_child_weight=1.0, num_leaves=int(2 ** 5 * 0.7),
                 n_estimators=2000, depth=5, seed=None):
        super().__init__(target_col)
        lgbm_params = {
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "subsample": subsample,
            "min_child_weight": min_child_weight,
            "num_leaves": num_leaves,
            "random_state": seed,
            "n_estimators": n_estimators,
            "max_depth": depth,
            "objective": "regression",
            "metrics": "rmse"
        }
        self.model = _LgbmRegressor(
            verbose_eval, early_stopping_rounds, lgbm_params)

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


class LgbmMaeRegressor(RegressorBase):
    """目的関数がMAEのlgbm回帰モデル."""

    def __init__(self, target_col, verbose_eval=100, early_stopping_rounds=100, colsample_bytree=0.8,
                 reg_alpha=0, reg_lambda=0, subsample=0.8, min_child_weight=1.0, num_leaves=int(2 ** 5 * 0.7),
                 n_estimators=2000, depth=5, seed=None):
        super().__init__(target_col)
        lgbm_params = {
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "subsample": subsample,
            "min_child_weight": min_child_weight,
            "num_leaves": num_leaves,
            "random_state": seed,
            "n_estimators": n_estimators,
            "max_depth": depth,
            "objective": "regression_l1",
            "metrics": "mae"
        }
        self.model = _LgbmRegressor(
            verbose_eval, early_stopping_rounds, lgbm_params)

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        valid_x, valid_y = self.split_df2xy(valid_df)
        return self.model.fit(train_x, train_y, valid_x, valid_y)

    def _predict(self, test_df):
        """test_dfにtarget_colが入っていても大丈夫."""
        return self.model.predict(test_df)

    def get_importance(self):
        """lgbmのみ追加で実装している."""
        return self.model.get_importance()

    def return_flag(self):
        return self.model.return_flag()
