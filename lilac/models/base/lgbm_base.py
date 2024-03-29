import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from category_encoders import OrdinalEncoder


class LgbmBase:
    """LGBMのベース.object型があればlabel encodingして元のカテゴリ変数は削除する"""

    def __init__(self, verbose_eval, early_stopping_rounds, n_estimators, lgbm_params):
        lgbm_params["num_leaves"] = int(2 ** lgbm_params["max_depth"] * 0.7)
        lgbm_params["verbose"] = -1
        self.lgbm_params = lgbm_params
        print(lgbm_params)
        self.verbose_eval = verbose_eval
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = n_estimators
        self.encoder = None
        self.model = None

    def get_object_cols(self, df):
        return df.select_dtypes(include=[object]).columns

    def fit(self, train_x, train_y, valid_x, valid_y):
        object_cols = self.get_object_cols(train_x)
        self.encoder = OrdinalEncoder()
        train_x_cat = self.encoder.fit_transform(train_x[object_cols]).add_suffix("_enc")
        valid_x_cat = self.encoder.transform(valid_x[object_cols]).add_suffix("_enc")
        train_x = pd.concat([train_x, train_x_cat], axis=1)
        valid_x = pd.concat([valid_x, valid_x_cat], axis=1)
        train_x = train_x.drop(object_cols, axis=1)
        valid_x = valid_x.drop(object_cols, axis=1)

        self.cols = list(train_x.columns)

        lgb_train, lgb_valid = self.get_dataset(train_x, train_y, valid_x, valid_y)
        self.model = lgb.train(
            params=self.lgbm_params,
            train_set=lgb_train,
            num_boost_round=self.num_boost_round,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(self.verbose_eval),
            ],
        )
        return self

    def get_dataset(self, train_x, train_y, valid_x, valid_y):
        lgb_train = lgb.Dataset(train_x, train_y)
        lgb_valid = lgb.Dataset(valid_x, valid_y, reference=lgb_train)
        return lgb_train, lgb_valid

    def get_importance(self):
        """特徴量の重要度を出力する."""
        return pd.DataFrame(
            self.model.feature_importance(importance_type="gain"), index=self.cols, columns=["importance"]
        ).to_dict()

    def return_flag(self):
        return "lgbm_" + "_".join([str(v) for v in self.lgbm_params.values()])


class LgbmClassifierBase(LgbmBase):
    """ベースとなるLGBMのclassifierモデル."""

    def __init__(self, verbose_eval, early_stopping_rounds, n_estimators, lgbm_params, class_weight):
        self.class_weight = class_weight
        super().__init__(verbose_eval, early_stopping_rounds, n_estimators, lgbm_params)

    def get_dataset(self, train_x, train_y, valid_x, valid_y):
        """lgb.train()のparamsにはclass_weightを入れられないみたいなので、Datasetにinstanceごとのweightを自前で計算して入れる."""

        def get_weight(y):
            class_weight = len(y) / np.bincount(y) * len(set(y))
            return [class_weight[e] for e in y]

        if self.class_weight is None:
            return super().get_dataset(train_x, train_y, valid_x, valid_y)

        if self.class_weight == "balanced":
            lgb_train = lgb.Dataset(train_x, train_y, weight=get_weight(train_y))
            lgb_valid = lgb.Dataset(valid_x, valid_y, weight=get_weight(valid_y), reference=lgb_train)
            return lgb_train, lgb_valid
        else:
            raise Exception(f"Invalid class_weight: {self.class_weight}")

    def predict_proba(self, test):
        object_cols = self.get_object_cols(test)
        test_cat = self.encoder.transform(test[object_cols]).add_suffix("_enc")
        test = pd.concat([test, test_cat], axis=1)
        test = test.drop(object_cols, axis=1)
        # LightGBMのBoosterはpredict_probaがないのでこれで予測
        # binary classifierなら1次元(クラス1の予測確率)
        # multi classifierならN次元の予測確率なはず
        raw_pred = self.model.predict(test)
        return raw_pred

    def return_flag(self):
        return f"{super().return_flag()}_cls"


class LgbmBinaryClassifierBase(LgbmClassifierBase):
    """LGBMのloglossで最適化するbin分類モデルのベース."""

    def __init__(self, verbose_eval, early_stopping_rounds, n_estimators, lgbm_params, class_weight):
        lgbm_params["objective"] = "binary"
        lgbm_params["metrics"] = "binary_logloss"
        super().__init__(verbose_eval, early_stopping_rounds, n_estimators, lgbm_params, class_weight)


class LgbmMultiClassifierBase(LgbmClassifierBase):
    """LGBMのloglossで最適化するmulti分類モデルのベース."""

    def __init__(self, verbose_eval, early_stopping_rounds, n_estimators, lgbm_params, class_weight):
        lgbm_params["objective"] = "multiclass"
        lgbm_params["metrics"] = "multi_logloss"
        super().__init__(verbose_eval, early_stopping_rounds, n_estimators, lgbm_params, class_weight)


class LgbmXentropyClassifierBase(LgbmClassifierBase):
    """LGBMのxentropyで最適化するbinary分類モデルのベース."""

    def __init__(self, verbose_eval, early_stopping_rounds, n_estimators, lgbm_params, class_weight):
        lgbm_params["objective"] = "xentropy"
        lgbm_params["metrics"] = "xentropy"
        super().__init__(verbose_eval, early_stopping_rounds, n_estimators, lgbm_params, class_weight)


class LgbmRegressorBase(LgbmBase):
    """ベースとなるLGBMの回帰モデル."""

    def predict(self, test):
        object_cols = self.get_object_cols(test)
        test_cat = self.encoder.transform(test[object_cols]).add_suffix("_enc")
        test = pd.concat([test, test_cat], axis=1)
        test = test.drop(object_cols, axis=1)
        raw_pred = self.model.predict(test)
        return raw_pred

    def return_flag(self):
        return f"{super().return_flag()}_reg"


class LgbmRmseRegressorBase(LgbmRegressorBase):
    """LGBMのRMSEで最適化する回帰モデルのベース."""

    def __init__(self, verbose_eval, early_stopping_rounds, n_estimators, lgbm_params):
        lgbm_params["objective"] = "regression"
        lgbm_params["metrics"] = "rmse"
        super().__init__(verbose_eval, early_stopping_rounds, n_estimators, lgbm_params)


class LgbmMaeRegressorBase(LgbmRegressorBase):
    """LGBMのMAEで最適化する回帰モデルのベース."""

    def __init__(self, verbose_eval, early_stopping_rounds, n_estimators, lgbm_params):
        lgbm_params["objective"] = "mae"
        lgbm_params["metrics"] = "mae"
        super().__init__(verbose_eval, early_stopping_rounds, n_estimators, lgbm_params)


class LgbmFairRegressorBase(LgbmRegressorBase):
    """LGBMのFairで最適化する回帰モデルのベース."""

    def __init__(self, verbose_eval, early_stopping_rounds, n_estimators, lgbm_params):
        lgbm_params["objective"] = "fair"
        lgbm_params["metrics"] = "mae"
        super().__init__(verbose_eval, early_stopping_rounds, n_estimators, lgbm_params)


class LgbmRmsleRegressorBase(LgbmRmseRegressorBase):
    """LGBMのRMSLEで最適化する回帰モデルのベース."""

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
