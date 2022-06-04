from lilac.models.base.catb_base import _CatbRegressor, _CatbRmsleRegressor
from lilac.models.model_base import RegressorBase


class CatbRmseRegressor(RegressorBase):
    """CatBoost RMSEで最適化"""

    def __init__(
        self,
        target_col,
        early_stopping_rounds=100,
        n_estimators=2000,
        depth=5,
        seed=None,
        learning_rate=0.1,
        random_strength=1,
        bagging_temperature=0.1,
        od_type="IncToDec",
        od_wait=10,
    ):
        super().__init__(target_col)
        catb_params = {
            "learning_rate": learning_rate,
            "random_strength": random_strength,
            "bagging_temperature": bagging_temperature,
            "od_type": od_type,
            "od_wait": od_wait,
            "random_seed": seed,
            "num_boost_round": n_estimators,
            "depth": depth,
        }
        self.model = _CatbRegressor("RMSE", early_stopping_rounds, catb_params)

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        valid_x, valid_y = self.split_df2xy(valid_df)
        self.model.fit(train_x, train_y, valid_x, valid_y)

    def _predict(self, test_df):
        return self.model.predict(test_df)

    def return_flag(self):
        return self.model.return_flag()


class CatbRmsleRegressor(RegressorBase):
    """CatBoost RMSLEで最適化"""

    def __init__(
        self,
        target_col,
        early_stopping_rounds=100,
        n_estimators=2000,
        depth=5,
        seed=None,
        learning_rate=0.1,
        random_strength=1,
        bagging_temperature=0.1,
        od_type="IncToDec",
        od_wait=10,
    ):
        super().__init__(target_col)
        catb_params = {
            "learning_rate": learning_rate,
            "random_strength": random_strength,
            "bagging_temperature": bagging_temperature,
            "od_type": od_type,
            "od_wait": od_wait,
            "random_seed": seed,
            "num_boost_round": n_estimators,
            "depth": depth,
        }
        self.model = _CatbRmsleRegressor(early_stopping_rounds, catb_params)

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        valid_x, valid_y = self.split_df2xy(valid_df)
        self.model.fit(train_x, train_y, valid_x, valid_y)

    def _predict(self, test_df):
        return self.model.predict(test_df)

    def return_flag(self):
        return f"{self.model.return_flag()}_rmsle"
