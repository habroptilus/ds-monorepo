import pandas as pd

from lilac.models.model_base import BinaryClassifierBase, RegressorBase


class XentropyRegressorBase(RegressorBase):
    """[0,1]にスケーリングしてcross entropyで学習するベースモデル."""

    def __init__(self, target_col, model):
        super().__init__(target_col)
        if issubclass(model.__class__, BinaryClassifierBase):
            ValueError(
                f"XentropyRegressorBase allows only models that implemented BinaryClassifierBase. {model.__class__}"
            )
        self.model = model

    def fit(self, train_df, valid_df):
        train_df = train_df.copy()
        valid_df = valid_df.copy()
        df = pd.concat([train_df, valid_df])
        self.min = df[self.target_col].min()
        self.max = df[self.target_col].max()
        train_df[self.target_col] = self.min_max_scaling(train_df[self.target_col])
        valid_df[self.target_col] = self.min_max_scaling(valid_df[self.target_col])

        return self.model.fit(train_df, valid_df)

    def min_max_scaling(self, y):
        return (y - self.min) / (self.max - self.min)

    def _predict(self, test_df):
        """予測した後元に戻す"""
        pred = self.model.predict_proba(test_df)
        return pred * (self.max - self.min) + self.min
