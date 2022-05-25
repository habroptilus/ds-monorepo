from lilac.features.generator_base import FeaturesBase
import pandas as pd


class LagFeatures(FeaturesBase):
    """ラグ特徴量を作成する.


    ts_colについてsort
    key_colについてgroupbyし
    してdiff, shift, pct_change, rollingを実施する
    sortの時に順番がオリジナルと変わっている可能性があるのでsort_indexして返す.
    """

    def __init__(self,  key_col, input_cols, ts_col, lags=None, rolling_windows=None, features_dir=None):
        self.key_col = key_col
        self.ts_col = ts_col
        self.input_cols = input_cols
        self.lags = lags or [1]
        self.rolling_windows = rolling_windows or [3]
        super().__init__(features_dir)

    def transform(self, df):
        df_list = []
        df = df.sort_values(self.ts_col)
        grp_df = df.groupby(self.key_col)[self.input_cols]
        for lag in self.lags:
            if lag == 0:
                continue
            # shift
            shift_df = grp_df.shift(lag).add_prefix(
                f"Shift{lag}_").sort_index()

            # diff
            diff_df = grp_df.diff(lag).add_prefix(f"Diff{lag}_").sort_index()

            # pct_change
            pct_df = grp_df.pct_change(lag).add_prefix(
                f"Pct{lag}_").sort_index()

            df_list.extend([shift_df, diff_df, pct_df])
        for window in self.rolling_windows:
            tmp_df = grp_df.rolling(window, min_periods=1)
            rolling_df = tmp_df.mean().add_prefix(
                f"Rolling{window}_mean_").sort_index().reset_index(drop=True)
            df_list.append(rolling_df)
        return pd.concat(df_list, axis=1)
