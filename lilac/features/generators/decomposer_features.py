import pandas as pd
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from umap import UMAP

from lilac.features.generator_base import FeaturesBase
from lilac.features.generators.features_pipeline import FeaturesPipeline
from lilac.features.generators.scaling_features import StandardScalingFeatures


class StandardizedDecomposer(FeaturesPipeline):
    """標準化をしてからPCAやUMAPを用いて次元削減する."""

    def __init__(self, decomposer_str, n_components, prefix, input_cols=None, random_state=None, features_dir=None):
        super().__init__(
            feature_generators=[
                StandardScalingFeatures(input_cols=input_cols, features_dir=features_dir),
                DecompositionFeatures(
                    decomposer_str=decomposer_str,
                    n_components=n_components,
                    prefix=prefix,
                    random_state=random_state,
                    features_dir=features_dir,
                ),
            ],
            use_prev_only=True,
            features_dir=features_dir,
        )


class DecompositionFeatures(FeaturesBase):
    def __init__(self, decomposer_str, n_components, prefix, input_cols=None, random_state=None, features_dir=None):
        """PCAやUMAP,SVD, NMFを用いて次元削減する.

        :n_components: 削減先の次元数.
        :input_cols: 適用するカラム.指定しないと全カラムになる.
        :prefix: カラム名が{prefix}_{decomposer_str}_1,...となる
        :random_state
        :features_dir
        """
        self.decomposer_str = decomposer_str
        self.input_cols = input_cols
        self.n_components = n_components
        self.random_state = random_state
        self.prefix = prefix
        super().__init__(features_dir)

    def fit(self, df):
        models = {"pca": PCA, "umap": UMAP, "svd": TruncatedSVD, "nmf": NMF}
        model = models.get(self.decomposer_str)
        if model is None:
            raise Exception(f"Invalid decomposer_str: '{self.decomposer_str}'")
        input_cols = self.resolve_input_cols(df)
        self.model = model(n_components=self.n_components, random_state=self.random_state)
        self.model.fit(df[input_cols])
        return self

    def transform(self, df):
        input_cols = self.resolve_input_cols(df)
        data = self.model.transform(df[input_cols])

        output_cols = [f"{self.prefix}_{self.decomposer_str}_{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(data, columns=output_cols)

    def resolve_input_cols(self, df):
        if self.input_cols is None:
            return df.columns
        else:
            return self.input_cols
