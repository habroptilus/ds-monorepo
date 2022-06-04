"""TargetEncoder.他の特徴量生成と異なる仕様なので別ディレクトリで管理."""
import numpy as np
import pandas as pd


class TargetEncoder:
    """TargetEncodingを行う.CrossValidationのfold分割と合わせて変換を一回で済ませるver.
    note: これでもout of foldは保たれているが、学習データ変換にvalidデータを使っている点がtestデータとの関係と異なるので
    気になる人はvalidを抜いて学習データ変換をする方がいいが、この場合はCrossValidationの分割数回異なる変換をかける事になる.

    単純なRMSE回帰や2値分類はこのまま適用できる.
    TODO : 多クラス分類->クラスごとに所属確率を使ってencoding
    """

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
        return_cols = []
        for input_col in self.input_cols:
            enc_col = f"{input_col}_{self.suffix}"
            result[enc_col] = None
            for tdx, vdx in self.folds_gen.run(df_copied):
                train, valid = df_copied.iloc[tdx], df_copied.iloc[vdx]
                target_mean = train.groupby(input_col)[self.target_col].mean()
                result.loc[vdx, enc_col] = valid[input_col].map(target_mean)
            return_cols.append(enc_col)

        self.target_means_for_test = {}
        # testは全体で計算
        for input_col in self.input_cols:
            self.target_means_for_test[input_col] = df.groupby(input_col)[self.target_col].mean()
        return result[return_cols]

    def transform(self, df):
        results = []
        return_cols = []
        for input_col, target_mean in self.target_means_for_test.items():
            results.append(df[input_col].map(target_mean))
            return_cols.append(f"{input_col}_{self.suffix}")
        result = pd.concat(results, axis=1)
        result.columns = return_cols
        return result
