from xfeat import ArithmeticCombinations, ConcatCombination

from lilac.features.generator_base import FeaturesBase


class CategoryCombination(FeaturesBase):
    """カテゴリ変数をxfeatのConcatCombinationを使って結合する."""

    def __init__(self, input_cols, r=2, output_suffix="_cc", features_dir=None):
        self.input_cols = input_cols
        self.r = r
        self.output_suffix = output_suffix
        super().__init__(features_dir)

    def transform(self, df):
        return ConcatCombination(drop_origin=True, output_suffix=self.output_suffix, r=self.r).fit_transform(
            df[self.input_cols].fillna("[NONE]")
        )


class NumericCombination(FeaturesBase):
    """数値特徴量をxfeatのArithmeticCombinationを使って結合する."""

    def __init__(self, input_cols, operator, r=2, features_dir=None):
        self.operators_suffix = {
            "+": "_add",
            "-": "_sub",
            "*": "_prod",
            "/": "_div",
        }
        if operator not in self.operators_suffix:
            raise Exception(f"Operator '{operator}' is invalid.")
        self.input_cols = input_cols
        self.operator = operator
        self.r = r
        super().__init__(features_dir)

    def transform(self, df):
        return ArithmeticCombinations(
            drop_origin=True, operator=self.operator, output_suffix=self.operators_suffix[self.operator], r=self.r
        ).fit_transform(df[self.input_cols])
