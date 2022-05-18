from tools.features.generator_base import FeaturesBase
from category_encoders import CountEncoder, OneHotEncoder, OrdinalEncoder


class CategoryEncoding(FeaturesBase):
    """category_encodersを使ったカテゴリ変数のエンコーディングを行う.input_colsを指定しないと自動でcategory型のカラムが適用先になる."""
    str2model = {
        "count": {"model":  CountEncoder, "suffix": "_ce"},
        "onehot": {"model":  OneHotEncoder, "suffix": "_ohe"},
        "ordinal": {"model":  OrdinalEncoder, "suffix": "_oe"}
    }

    def __init__(self, encoder_str, input_cols=None, features_dir=None):
        self.encoder_str = encoder_str
        self.input_cols = input_cols
        super().__init__(features_dir)

    def fit(self, df):
        if self.encoder_str not in self.str2model:
            raise Exception(f"Invalid encoder_str '{self.encoder_str}'.")

        self.suffix = self.str2model[self.encoder_str]["suffix"]
        if self.input_cols is None:
            self.input_cols = df.select_dtypes(include=[object]).columns
        self.encoder = self.str2model[self.encoder_str]["model"](
            cols=self.input_cols)
        self.encoder.fit(df[self.input_cols])
        return self

    def transform(self, df):
        df = self.encoder.transform(df[self.input_cols])
        return df.add_suffix(self.suffix)
