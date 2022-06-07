"""catboost classfiers."""
from lilac.models import consts
from lilac.models.base.catb_base import _CatbBinaryClassfier, _CatbMultiClassfier
from lilac.models.model_base import BinaryClassifierBase, MultiClassifierBase


class CatbBinaryClassifier(BinaryClassifierBase):
    """目的関数がLoglossのcatboost2値分類モデル."""

    def __init__(
        self,
        target_col,
        early_stopping_rounds=consts.early_stopping_rounds,
        n_estimators=consts.n_estimators,
        depth=consts.n_estimators,
        seed=consts.seed,
        learning_rate=consts.learning_rate,
        random_strength=consts.random_strength,
        bagging_temperature=consts.bagging_temperature,
        od_type=consts.od_type,
        od_wait=consts.od_wait,
        class_weight=consts.class_weight,
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
        self.model = _CatbBinaryClassfier(early_stopping_rounds, catb_params, class_weight)

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        valid_x, valid_y = self.split_df2xy(valid_df)
        return self.model.fit(train_x, train_y, valid_x, valid_y)

    def _predict_proba(self, test_df):
        """test_dfにtarget_colが入っていても大丈夫."""
        return self.model.predict_proba(test_df)

    def return_flag(self):
        return self.model.return_flag()


class CatbMultiClassifier(MultiClassifierBase):
    """目的関数がLoglossのcatboost多値分類モデル."""

    def __init__(
        self,
        target_col,
        early_stopping_rounds=consts.early_stopping_rounds,
        n_estimators=consts.n_estimators,
        depth=consts.n_estimators,
        seed=consts.seed,
        learning_rate=consts.learning_rate,
        random_strength=consts.random_strength,
        bagging_temperature=consts.bagging_temperature,
        od_type=consts.od_type,
        od_wait=consts.od_wait,
        class_weight=consts.class_weight,
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
        self.model = _CatbMultiClassfier(early_stopping_rounds, catb_params, class_weight)

    def fit(self, train_df, valid_df):
        train_x, train_y = self.split_df2xy(train_df)
        valid_x, valid_y = self.split_df2xy(valid_df)
        return self.model.fit(train_x, train_y, valid_x, valid_y)

    def _predict_proba(self, test_df):
        """test_dfにtarget_colが入っていても大丈夫."""
        return self.model.predict_proba(test_df)

    def return_flag(self):
        return self.model.return_flag()
