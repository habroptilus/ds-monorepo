"""TargetEncoder.他の特徴量生成と異なる仕様なので別ディレクトリで管理."""
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


class _TargetEncoder:
    """TargetEncodingを行う. 自作のfolds_genを使える.なんかスコアが上がらないのでどこかバグっているのかも"""

    def __init__(self, input_cols, target_col, folds_gen, suffix="te", log_target=False):
        self.input_cols = input_cols
        self.target_col = target_col
        self.suffix = suffix
        self.folds_gen = folds_gen
        self.log_target = log_target

    def fit_transform(self, df):
        df_copied = df.copy()
        # RMSLEが評価指標の時に使用する.
        if self.log_target:
            df_copied[self.target_col] = np.log1p(df_copied[self.target_col])
        result = df.copy()
        for input_col in self.input_cols:
            enc_col = f"{input_col}_{self.suffix}"
            result[enc_col] = None
            for tdx, vdx in self.folds_gen.run(df_copied):
                train, valid = df_copied.iloc[tdx], df_copied.iloc[vdx]
                target_mean = train.groupby(input_col)[self.target_col].mean()
                result.loc[vdx, enc_col] = valid[input_col].map(target_mean)

        self.target_means_for_test = {}
        # testは全体で計算
        for input_col in self.input_cols:
            self.target_means_for_test[input_col] = df.groupby(input_col)[self.target_col].mean()
        return result

    def transform(self, df):
        df = df.copy()
        for input_col, target_mean in self.target_means_for_test.items():
            df[f"{input_col}_te"] = df[input_col].map(target_mean)
        return df
