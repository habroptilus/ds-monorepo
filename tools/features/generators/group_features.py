import pandas as pd
from xfeat import aggregation
from tools.features.generator_base import FeaturesBase


class GroupFeatures(FeaturesBase):
    """集約特徴量を計算する.fitしてtransformするタイプ.

    mean,max,min,median,sum,count,max-min,q75-q25、およびそれぞれの統計量と対象の数値のdiffやratio,diff_ratioを返す(diff,ratio,diff_ratioは一部のみ).
    なので返す特徴量の個数はlen(input_cols)*(5+3*5)となる.
    """

    def __init__(self, input_cols, group_key,  features_dir=None):
        self.group_key = group_key
        self.input_cols = input_cols
        super().__init__(features_dir)

    def fit(self, df):
        df, aggregated_cols = aggregation(df,
                                          group_key=self.group_key,
                                          group_values=self.input_cols,
                                          agg_methods=["mean", "max", "min",
                                                       "median", "sum"]
                                          )

        self._agg = df[~df.duplicated(subset=self.group_key)][[
            self.group_key]+aggregated_cols]
        return self

    def transform(self, df):
        df = pd.merge(df, self._agg, on=self.group_key, how="left")
        aggregated_cols = list(self._agg.columns)
        for agg_func in ["mean", "max", "min",
                         "median", "sum"]:
            df, result_cols = self.add_diff_ratio(df, agg_func)
            aggregated_cols.extend(result_cols)
        return df[aggregated_cols].drop(self.group_key, axis=1)

    def add_diff_ratio(self, data, agg_func):
        result_cols = []
        for group_col in self.input_cols:
            agg_col = f"agg_{agg_func}_{group_col}_grpby_{self.group_key}"
            data[f"{agg_col}_diff"] = data[group_col].values - \
                data[agg_col].values
            data[f"{agg_col}_ratio"] = data[group_col].values / \
                (data[agg_col]+1e-9)
            data[f"{agg_col}_diff_ratio"] = data[f"{agg_col}_diff"].values / \
                (data[agg_col]+1e-9)
            result_cols.extend(
                [f"{agg_col}_diff", f"{agg_col}_ratio", f"{agg_col}_diff_ratio"])
        return data, result_cols

    class MaxMin:
        def __call__(self, x):
            return x.max()-x.min()

        def __str__(self):
            return "max_min"

    class Q75Q25:
        def __call__(self, x):
            return x.quantile(0.75) - x.quantile(0.25)

        def __str__(self):
            return "q75_q25"
