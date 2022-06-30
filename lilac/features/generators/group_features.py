import pandas as pd
from xfeat import aggregation

from lilac.features.generator_base import FeaturesBase
from lilac.features.wrappers.brute_force_features import BruteForceFeatures
from lilac.features.wrappers.features_pipeline import FeaturesPipeline

default_agg_func_list = ["mean", "max", "min", "median", "sum", "std"]


class MeanDiffRatioGroupFeatures(FeaturesPipeline):
    """1段目で平均値のdiff, ratioを計算し、2段目でそれらの集約値のみ取り出す."""

    def __init__(self, group_key, input_cols, agg_func_list=None, features_dir=None):
        agg_func_list = agg_func_list or default_agg_func_list
        diff_ratio_cols = [
            f"agg_mean_{input_col}_grpby_{group_key}_{suffix}"
            for input_col in input_cols
            for suffix in ["diff", "ratio"]
        ]
        feature_generators = [
            GroupFeatures(
                group_keys=[group_key],
                input_cols=input_cols,
                agg_func_list=["mean"],
                do_add_diff=True,
                do_add_ratio=True,
                features_dir=features_dir,
            ),
            GroupFeatures(
                group_keys=[group_key],
                input_cols=diff_ratio_cols,
                agg_func_list=agg_func_list,
                do_add_diff=False,
                do_add_ratio=False,
                features_dir=features_dir,
            ),
        ]
        super().__init__(feature_generators=feature_generators, use_prev_only=False, features_dir=features_dir)


class GroupFeatures(BruteForceFeatures):
    """集約特徴量を作成する."""

    def __init__(
        self,
        group_keys,
        input_cols,
        do_add_diff=True,
        do_add_ratio=False,
        features_dir=None,
        agg_func_list=None,
    ):
        """
        group_keys: group byするカラム. 複数指定した場合、col_aかつcol_bの粒度でgroupbyするのではなく、独立に集約するので注意.
        input_cols: 集約される側
        agg_func_list: mean,max,min,median,sum
        do_add_diff: 集約値からの差を追加する
        do_add_ratio: 集約値からの比を追加する
        """

        agg_func_list = agg_func_list or default_agg_func_list
        params = {"input_col": input_cols, "agg_func": agg_func_list, "group_key": group_keys}
        super().__init__(
            params=params,
            FeaturesClass=_GroupFeatures,
            do_add_diff=do_add_diff,
            do_add_ratio=do_add_ratio,
            features_dir=features_dir,
        )


class _GroupFeatures(FeaturesBase):
    """(group_key, agg_func, input_col)の組に対して集約特徴量を計算する."""

    def __init__(self, input_col, agg_func, group_key, do_add_diff=True, do_add_ratio=False, features_dir=None):
        self.group_key = group_key
        self.input_col = input_col
        self.do_add_diff = do_add_diff
        self.do_add_ratio = do_add_ratio

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

        if self.do_add_ratio:
            df, result_cols = self.add_ratio(df)
            aggregated_cols.extend(result_cols)
        return df[aggregated_cols].drop(self.group_key, axis=1)

    def add_diff(self, data):
        """集約値からの差を計算する。"""
        result_cols = []
        added_df = pd.DataFrame()

        agg_col = f"agg_{self.agg_func}_{self.input_col}_grpby_{self.group_key}"
        added_df[f"{agg_col}_diff"] = data[self.input_col].values - data[agg_col].values
        result_cols.append(f"{agg_col}_diff")
        data = pd.concat([data, added_df], axis=1)
        return data, result_cols

    def add_ratio(self, data):
        """集約値からの比を計算する。"""
        result_cols = []
        added_df = pd.DataFrame()

        agg_col = f"agg_{self.agg_func}_{self.input_col}_grpby_{self.group_key}"
        added_df[f"{agg_col}_ratio"] = data[self.input_col].values / (data[agg_col].values + 1e-9)
        result_cols.append(f"{agg_col}_ratio")
        data = pd.concat([data, added_df], axis=1)
        return data, result_cols
