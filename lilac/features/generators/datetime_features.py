import pandas as pd
from lilac.features.generator_base import FeaturesBase


class DatetimeFeatures(FeaturesBase):
    """文字列型のdateのカラムをparseしyear,month,day,timestampを取得する.

    :yyyy-mm-ddの形式で正常に動作していることは確認した
    :欠損していても大丈夫.
    """

    def __init__(self, input_col=None, features_dir=None):
        self.input_col = input_col
        super().__init__(features_dir)

    def transform(self, df):
        df[f"{self.input_col}_datetime"] = pd.to_datetime(df[self.input_col])

        # datetimeからtimestampに
        # 大きすぎるので10**9で割った値を入れている
        df[f"{self.input_col}_ts"] = df[f"{self.input_col}_datetime"].apply(
            lambda x: x.timestamp()/1e9 if not isinstance(x, type(pd.NaT)) else None)

        # datetimeから年月日を計算
        df[f"{self.input_col}_year"] = df[f"{self.input_col}_datetime"].apply(
            lambda x: x.year if not isinstance(x, type(pd.NaT)) else None)
        df[f"{self.input_col}_month"] = df[f"{self.input_col}_datetime"].apply(
            lambda x: x.month if not isinstance(x, type(pd.NaT)) else None).astype(str)
        df[f"{self.input_col}_day"] = df[f"{self.input_col}_datetime"].apply(
            lambda x: x.day if not isinstance(x, type(pd.NaT)) else None).astype(str)
        return df[[f"{self.input_col}_{suffix}" for suffix in ["year", "month", "day", "ts"]]]
