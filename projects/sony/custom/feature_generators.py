from lilac.features.generator_base import FeaturesBase
from lilac.features.wrappers.features_pipeline import FeaturesPipeline
from lilac.features.generators.scaling_features import StandardScalingFeatures


class AverageGasStandardScaledStats(FeaturesPipeline):
    """各気体の統計量を標準化し、統計量の種類ごとに平均する.Pipelineを使った実装の例としても参照されたい."""

    prefix_list = ["co", "no2", "so2", "o3"]
    suffix_list = ["cnt", "min", "max", "rng", "var"]

    def __init__(self, features_dir=None):
        input_cols = [f"{prefix}_{suffix}" for prefix in self.prefix_list for suffix in self.suffix_list]
        super().__init__(
            feature_generators=[
                StandardScalingFeatures(input_cols, features_dir=features_dir),
                AverageGasStats(features_dir),
            ],
            use_prev_only=False,
            features_dir=features_dir,
        )


class AverageGasStats(FeaturesBase):
    """各気体の統計量を統計量の種類ごとに平均する."""

    prefix_list = ["co", "no2", "so2", "o3"]
    suffix_list = ["cnt", "min", "max", "rng", "var"]

    def transform(self, df):
        input_cols_dict = {
            suffix: [f"{prefix}_{suffix}_std_scaled" for prefix in self.prefix_list] for suffix in self.suffix_list
        }
        for suffix, input_cols in input_cols_dict.items():
            df[f"{suffix}_mean"] = df[input_cols].mean(axis=1)
        return df[[f"{suffix}_mean" for suffix in self.suffix_list]]
