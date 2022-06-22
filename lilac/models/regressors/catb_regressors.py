from lilac.models import consts
from lilac.models.base.catb_base import CatbMaeRegressorBase, CatbRmseRegressorBase, CatbRmsleRegressorBase
from lilac.models.model_base import RegressorBase


class CatbRegressor(RegressorBase):
    """CatBoost 回帰モデル."""

    def __init__(
        self,
        target_col,
        base_model,
        early_stopping_rounds=consts.early_stopping_rounds,
        n_estimators=consts.n_estimators,
        depth=consts.depth,
        seed=consts.seed,
        learning_rate=consts.learning_rate,
        random_strength=consts.random_strength,
        bagging_temperature=consts.bagging_temperature,
        od_type=consts.od_type,
        od_wait=consts.od_wait,
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
        self.model = base_model(early_stopping_rounds, catb_params)

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        valid_x, valid_y = self.split_df2xy(valid_df)
        self.model.fit(train_x, train_y, valid_x, valid_y)

    def _predict(self, test_df):
        return self.model.predict(test_df)

    def return_flag(self):
        return self.model.return_flag()


class CatbRmseRegressor(CatbRegressor):
    """CatBoost RMSEで最適化"""

    def __init__(
        self,
        target_col,
        early_stopping_rounds=consts.early_stopping_rounds,
        n_estimators=consts.n_estimators,
        depth=consts.depth,
        seed=consts.seed,
        learning_rate=consts.learning_rate,
        random_strength=consts.random_strength,
        bagging_temperature=consts.bagging_temperature,
        od_type=consts.od_type,
        od_wait=consts.od_wait,
    ):
        super().__init__(
            target_col=target_col,
            base_model=CatbRmseRegressorBase,
            early_stopping_rounds=early_stopping_rounds,
            n_estimators=n_estimators,
            depth=depth,
            seed=seed,
            learning_rate=learning_rate,
            random_strength=random_strength,
            bagging_temperature=bagging_temperature,
            od_type=od_type,
            od_wait=od_wait,
        )


class CatbMaeRegressor(CatbRegressor):
    """CatBoost MAEで最適化"""

    def __init__(
        self,
        target_col,
        early_stopping_rounds=consts.early_stopping_rounds,
        n_estimators=consts.n_estimators,
        depth=consts.depth,
        seed=consts.seed,
        learning_rate=consts.learning_rate,
        random_strength=consts.random_strength,
        bagging_temperature=consts.bagging_temperature,
        od_type=consts.od_type,
        od_wait=consts.od_wait,
    ):
        super().__init__(
            target_col=target_col,
            base_model=CatbMaeRegressorBase,
            early_stopping_rounds=early_stopping_rounds,
            n_estimators=n_estimators,
            depth=depth,
            seed=seed,
            learning_rate=learning_rate,
            random_strength=random_strength,
            bagging_temperature=bagging_temperature,
            od_type=od_type,
            od_wait=od_wait,
        )


class CatbRmsleRegressor(CatbRegressor):
    """CatBoost Rmsleで最適化"""

    def __init__(
        self,
        target_col,
        early_stopping_rounds=consts.early_stopping_rounds,
        n_estimators=consts.n_estimators,
        depth=consts.depth,
        seed=consts.seed,
        learning_rate=consts.learning_rate,
        random_strength=consts.random_strength,
        bagging_temperature=consts.bagging_temperature,
        od_type=consts.od_type,
        od_wait=consts.od_wait,
    ):
        super().__init__(
            target_col=target_col,
            base_model=CatbRmsleRegressorBase,
            early_stopping_rounds=early_stopping_rounds,
            n_estimators=n_estimators,
            depth=depth,
            seed=seed,
            learning_rate=learning_rate,
            random_strength=random_strength,
            bagging_temperature=bagging_temperature,
            od_type=od_type,
            od_wait=od_wait,
        )
