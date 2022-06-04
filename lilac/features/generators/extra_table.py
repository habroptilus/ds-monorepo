import pandas as pd

from lilac.features.generator_base import FeaturesBase


class ExtraTableJoin(FeaturesBase):
    """外部テーブルを結合する."""

    def __init__(self, csv_path, join_on, features_dir=None):
        self.csv_path = csv_path
        self.join_on = join_on
        super().__init__(features_dir)

    def transform(self, df):
        ex_table = pd.read_csv(self.csv_path)
        df = pd.merge(df, ex_table, on=self.join_on, how="left")
        return_cols = list(ex_table.columns)
        if self.join_on in return_cols:
            return_cols.remove(self.join_on)
        return df[return_cols]
