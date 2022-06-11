import pandas as pd
from xfeat import aggregation

from lilac.features.generator_base import FeaturesBase
from lilac.features.wrappers.brute_force_features import BruteForceFeatures


class GroupFeatures(BruteForceFeatures):
    def __init__(self, group_key, input_cols, do_add_diff=True, features_dir=None, agg_func_list=None):
        default_agg_func_list = ["mean", "max", "min", "median", "sum"]
        if agg_func_list and len(set(agg_func_list) - set(default_agg_func_list)) > 0:
            raise Exception(f"Invalid agg_func are found '{set(agg_func_list) - set(default_agg_func_list)}'")
        agg_func_list = agg_func_list or default_agg_func_list
        super().__init__(
            params_a=input_cols,
            params_b=agg_func_list,
            name_a="input_col",
            name_b="agg_func",
            FeaturesClass=_GroupFeatures,
            group_key=group_key,
            do_add_diff=do_add_diff,
            features_dir=features_dir,
        )


class _GroupFeatures(FeaturesBase):
    """(agg_func,input_col)の組に対して集約特徴量を計算する."""

    def __init__(self, input_col, agg_func, group_key, do_add_diff=True, features_dir=None):
        self.group_key = group_key
        self.input_col = input_col
        self.do_add_diff = do_add_diff
        default_agg_func_list = ["mean", "max", "min", "median", "sum"]
        if agg_func not in default_agg_func_list:
            raise Exception(f"Invalid agg_func '{agg_func}'")
        self.agg_func = agg_func
        super().__init__(features_dir)

    def fit(self, df):
        df, aggregated_cols = aggregation(
            df, group_key=self.group_key, group_values=[self.input_col], agg_methods=[self.agg_func]
        )

        self._agg = df[~df.duplicated(subset=self.group_key)][[self.group_key] + aggregated_cols]
        return self

    def transform(self, df):
        df = pd.merge(df, self._agg, on=self.group_key, how="left")
        aggregated_cols = list(self._agg.columns)
        if self.do_add_diff:
            df, result_cols = self.add_diff(df)
            aggregated_cols.extend(result_cols)
        return df[aggregated_cols].drop(self.group_key, axis=1)

    def add_diff(self, data):
        """集約値からの差を計算する。

        ratioやdiff_ratioも計算してみたが、悪化することが多かったため削除した.
        """
        result_cols = []
        added_df = pd.DataFrame()

        agg_col = f"agg_{self.agg_func}_{self.input_col}_grpby_{self.group_key}"
        added_df[f"{agg_col}_diff"] = data[self.input_col].values - data[agg_col].values
        result_cols.append(f"{agg_col}_diff")
        data = pd.concat([data, added_df], axis=1)
        return data, result_cols
