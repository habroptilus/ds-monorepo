import pandas as pd
from lilac.features.generator_base import FeaturesBase


class DatetimeFeatures(FeaturesBase):
    """文字列型のdateのカラムをparseしyear,month,day,timestamp,曜日,土日かどうかを取得する.

    :yyyy-mm-ddの形式で正常に動作していることは確認した
    :欠損していても大丈夫.
    """

    def __init__(self, input_col=None, include_ymd=True, format_str='%Y%m%d', features_dir=None):
        self.input_col = input_col
        self.format_str = format_str
        self.include_ymd = include_ymd
        super().__init__(features_dir)

    def transform(self, df):
        df[f"{self.input_col}_dt"] = pd.to_datetime(
            df[self.input_col], format=self.format_str)

        # datetimeからtimestampに
        # 大きすぎるので10**9で割った値を入れている
        df[f"{self.input_col}_ts"] = df[f"{self.input_col}_dt"].apply(
            lambda x: x.timestamp()/1e9 if not isinstance(x, type(pd.NaT)) else None)

        # datetimeから年月日を計算
        if self.include_ymd:
            df[f"{self.input_col}_year"] = df[f"{self.input_col}_dt"].apply(
                lambda x: x.year if not isinstance(x, type(pd.NaT)) else None)
            df[f"{self.input_col}_month"] = df[f"{self.input_col}_dt"].apply(
                lambda x: x.month if not isinstance(x, type(pd.NaT)) else None).astype(str)
            df[f"{self.input_col}_day"] = df[f"{self.input_col}_dt"].apply(
                lambda x: x.day if not isinstance(x, type(pd.NaT)) else None).astype(str)
        df[f"{self.input_col}_day_name"] = df[f"{self.input_col}_dt"].apply(
            lambda x: x.day_name if not isinstance(x, type(pd.NaT)) else None).astype(str)

        def get_is_weekend(x):
            if isinstance(x, type(pd.NaT)):
                return None
            elif x.dayofweek >= 5:
                return 1
            else:
                return 0

        df[f"{self.input_col}_is_weekend"] = df[f"{self.input_col}_dt"].apply(
            get_is_weekend).astype(str)
        suffixes = ["ts", "day_name", "is_weekend"]
        if self.include_ymd:
            suffixes.extend(["year", "month", "day"])
        return df[[f"{self.input_col}_{suffix}" for suffix in suffixes]]
