import pandas as pd
from sklearn.preprocessing import StandardScaler

from lilac.features.generator_base import FeaturesBase


class StandardScalingFeatures(FeaturesBase):
    """標準化をする."""

    def __init__(self, input_cols, features_dir=None):
        self.input_cols = input_cols
        super().__init__(features_dir)

    def fit(self, df):
        self.input_cols = self.resolve_input_cols(df)
        self.scaler = StandardScaler()
        self.scaler.fit(df[self.input_cols])
        return self

    def transform(self, df):
        data = self.scaler.transform(df[self.input_cols])
        return pd.DataFrame(data, columns=[f"{col}_std_scaled" for col in self.input_cols])

    def resolve_input_cols(self, df):
        if self.input_cols is None:
            return df.columns
        else:
            return self.input_cols
