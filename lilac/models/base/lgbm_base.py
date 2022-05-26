import lightgbm as lgb
import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder


class _LgbmBase:
    """LGBMのベース.object型があればlabel encodingして元のカテゴリ変数は削除する"""

    def __init__(self, verbose_eval, early_stopping_rounds, lgbm_params):
        self.lgbm_params = lgbm_params
        self.verbose_eval = verbose_eval
        self.early_stopping_rounds = early_stopping_rounds
        self.encoder = OrdinalEncoder()
        self.model = self.get_model()

    def get_model(self):
        raise Exception("Implement please.")

    def fit(self, train_x, train_y, valid_x, valid_y):
        self.object_cols = train_x.select_dtypes(include=[object]).columns
        train_x_cat = self.encoder.fit_transform(train_x[self.object_cols]).add_suffix("_enc")
        valid_x_cat = self.encoder.transform(valid_x[self.object_cols]).add_suffix("_enc")
        train_x = pd.concat([train_x, train_x_cat], axis=1)
        valid_x = pd.concat([valid_x, valid_x_cat], axis=1)
        train_x = train_x.drop(self.object_cols, axis=1)
        valid_x = valid_x.drop(self.object_cols, axis=1)

        self.cols = list(train_x.columns)
        self.model.fit(
            train_x,
            train_y,
            eval_set=[(train_x, train_y), (valid_x, valid_y)],
            eval_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=True),
                lgb.log_evaluation(self.verbose_eval),
            ],
        )

        return self

    def get_importance(self):
        """特徴量の重要度を出力する."""
        return pd.DataFrame(self.model.feature_importances_, index=self.cols, columns=["importance"])

    def return_flag(self):
        return "lgbm_" + "_".join([str(v) for v in self.lgbm_params.values()])


class _LgbmClassifier(_LgbmBase):
    """ベースとなるLGBMのclassifierモデル."""

    def __init__(self, verbose_eval, early_stopping_rounds, lgbm_params, class_weight):
        lgbm_params["class_weight"] = class_weight
        super().__init__(verbose_eval, early_stopping_rounds, lgbm_params)

    def get_model(self):
        return lgb.LGBMClassifier(**self.lgbm_params)

    def predict_proba(self, test):
        test_cat = self.encoder.transform(test[self.object_cols]).add_suffix("_enc")
        test = pd.concat([test, test_cat], axis=1)
        test = test.drop(self.object_cols, axis=1)
        raw_pred = self.model.predict_proba(test, num_iteration=self.model.best_iteration_)
        return raw_pred

    def return_flag(self):
        return f"{super().return_flag()}_cls"


class _LgbmRegressor(_LgbmBase):
    """ベースとなるLGBMの回帰モデル."""

    def get_model(self):
        return lgb.LGBMRegressor(**self.lgbm_params)

    def predict(self, test):
        test_cat = self.encoder.transform(test[self.object_cols]).add_suffix("_enc")
        test = pd.concat([test, test_cat], axis=1)
        test = test.drop(self.object_cols, axis=1)
        raw_pred = self.model.predict(test, num_iteration=self.model.best_iteration_)
        return raw_pred

    def return_flag(self):
        return f"{super().return_flag()}_reg"


class _LgbmRmsleRegressor(_LgbmRegressor):
    """LGBMのRMSLEで最適化する回帰モデル."""

    def __init__(self, verbose_eval, early_stopping_rounds, lgbm_params):
        lgbm_params["objective"] = "regression"
        lgbm_params["metrics"] = "rmse"
        super().__init__(verbose_eval, early_stopping_rounds, lgbm_params)

    def fit(self, train_x, train_y, valid_x, valid_y):
        # yをlog変換
        train_y = self._pre_process_y(train_y)
        valid_y = self._pre_process_y(valid_y)
        # あとはもとのと一緒
        return super().fit(train_x, train_y, valid_x, valid_y)

    def predict(self, test):
        # 普通に予測
        raw_pred = super().predict(test)
        # yを戻す
        return self._post_process_y(raw_pred)

    def _pre_process_y(self, y):
        """RMSEのlgbmでRMSLEを使った学習を行うために、前処理を行う.

        :yはlog変換する
        """
        return np.log1p(y)

    def _post_process_y(self, pred):
        """RMSEのlgbmでRMSLEを使った学習を行うために、後処理を行う.
        対数をとった分を戻すのと、負の数が出力されることを防ぐために0と比較したmaxをとる.
        """
        return np.maximum(np.exp(pred) - 1, 0)