"""行方向に何かを行う系の特徴量."""
from lilac.features.generator_base import FeaturesBase


class NullColumnsNum(FeaturesBase):
    """nullの個数を計算する."""

    def transform(self, df):
        df["null_count"] = df.isnull().sum(axis=1)
        return df[["null_count"]]
