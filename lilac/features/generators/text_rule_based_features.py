"""テキストを対象としたルールベースの特徴量."""
from lilac.features.generator_base import FeaturesBase


class TextLengthFeature(FeaturesBase):
    """単語の数を数える."""

    def __init__(self, input_col, sep=" ", features_dir=None):
        self.input_col = input_col
        self.sep = sep
        super().__init__(features_dir=features_dir)

    def transform(self, df):
        output_col = f"{self.input_col}_text_length"
        df[output_col] = df[self.input_col].apply(lambda x: len(x.split(self.sep)))
        return df[[output_col]]
