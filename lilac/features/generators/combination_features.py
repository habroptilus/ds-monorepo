from xfeat import ConcatCombination

from lilac.features.generator_base import FeaturesBase


class CategoryCombination(FeaturesBase):
    """カテゴリ変数をxfeatのConcatCombinationを使って結合する."""

    def __init__(self, input_cols, r=2, output_suffix="_cc", features_dir=None):
        self.input_cols = input_cols
        self.r = r
        self.output_suffix = output_suffix
        super().__init__(features_dir)

    def transform(self, df):
        original_cols = set(df.columns)
        df = ConcatCombination(output_suffix=self.output_suffix, r=self.r).fit_transform(
            df[self.input_cols].fillna("[NONE]")
        )
        return_cols = list(set(df.columns) - original_cols)
        return df[return_cols]
