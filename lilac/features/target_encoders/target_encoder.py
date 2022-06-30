"""TargetEncoder.他の特徴量生成と異なる仕様なので別ディレクトリで管理."""
import pickle

import numpy as np
import xfeat
from sklearn.model_selection import KFold


class TargetEncoder:
    """xfeatのTargetEncoderのラッパー.Kfoldしか使えない.元のカテゴリは残す.

    単純なRMSE回帰や2値分類はこのまま適用できる.
    TODO : 多クラス分類->クラスごとに所属確率を使ってencoding
    """

    def __init__(self, input_cols, target_col, random_state=None, n_split=5, shuffle=True, log_target=False):
        self.input_cols = input_cols
        self.target_col = target_col
        self.log_target = log_target
        self.shuffle = shuffle
        self.n_split = n_split
        self.random_state = random_state
        self.encoder = None

    def fit_transform(self, df):
        """logをとるならとって、xfeat.TargetEncoderのfit_transformを呼ぶ"""
        target_col = self.target_col
        if self.log_target:
            target_col = f"{self.target_col}_log"
            df[target_col] = np.log1p(df[self.target_col])

        folds_gen = KFold(n_splits=self.n_split, random_state=self.random_state, shuffle=self.shuffle)
        self.encoder = xfeat.TargetEncoder(target_col=target_col, input_cols=self.input_cols, fold=folds_gen)
        result = self.encoder.fit_transform(df)

        if self.log_target:
            result = result.drop(target_col, axis=1)
        return result

    def transform(self, df):
        return self.encoder.transform(df)

    def save(self, filepath):
        if self.encoder is None:
            raise Exception("Saving encoder before training cannot be done. Please train encoder first.")
        pickle.dump(self.encoder, open(filepath, "wb"))

    def load(self, filepath):
        if self.encoder:
            raise Exception("Trained encoder already exists.")
        self.encoder = pickle.load(open(filepath, "rb"))
