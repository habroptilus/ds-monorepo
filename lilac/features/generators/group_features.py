import pandas as pd
from xfeat import aggregation

from lilac.features.generator_base import FeaturesBase


class GroupFeatures(FeaturesBase):
    """集約特徴量を計算する.fitしてtransformするタイプ.

    mean,max,min,median,sumおよびそれぞれの統計量と対象の数値のdiffを計算する(後者はdo_add_diff=Trueの場合のみ)
    なので返す特徴量の個数はlen(input_cols)*(5+(1))となる.
    """

    def __init__(self, input_cols, group_key, do_add_diff=True, features_dir=None):
        self.group_key = group_key
        self.input_cols = input_cols
        self.do_add_diff = do_add_diff
        self.agg_func_list = ["mean", "max", "min", "median", "sum"]
        super().__init__(features_dir)

    def fit(self, df):
        df, aggregated_cols = aggregation(
            df, group_key=self.group_key, group_values=self.input_cols, agg_methods=self.agg_func_list
        )

        self._agg = df[~df.duplicated(subset=self.group_key)][[self.group_key] + aggregated_cols]
        return self

    def transform(self, df):
        df = pd.merge(df, self._agg, on=self.group_key, how="left")
        aggregated_cols = list(self._agg.columns)
        if self.do_add_diff:
            for agg_func in self.agg_func_list:
                df, result_cols = self.add_diff(df, agg_func)
                aggregated_cols.extend(result_cols)
        return df[aggregated_cols].drop(self.group_key, axis=1)

    def add_diff(self, data, agg_func):
        """集約値からの差を計算する。

        ratioやdiff_ratioも計算してみたが、悪化することが多かったため削除した.
        """
        result_cols = []
        added_df = pd.DataFrame()
        for group_col in self.input_cols:
            agg_col = f"agg_{agg_func}_{group_col}_grpby_{self.group_key}"
            added_df[f"{agg_col}_diff"] = data[group_col].values - data[agg_col].values
            result_cols.append(f"{agg_col}_diff")
        data = pd.concat([data, added_df], axis=1)
        return data, result_cols
