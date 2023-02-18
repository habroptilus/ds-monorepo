import pandas as pd

from lilac.features.generator_base import FeaturesBase
from lilac.features.generators.decomposer_features import DecompositionFeatures, StandardizedDecomposer
from lilac.features.nlp.text_vectorizers.text_vectorizer_factory import TextVectorizerFactory
from lilac.features.wrappers.features_pipeline import FeaturesPipeline


class SentenceVectorFeature(FeaturesBase):
    """textをベクトル化して特徴量を作成する.

    内部的にはnlpモジュールのtext_vectorizerを使う.
    """

    def __init__(self, input_col, vectorizer_str, vectorizer_params=None, features_dir=None):
        self.input_col = input_col
        self.vectorizer_str = vectorizer_str
        self.vectorizer_params = vectorizer_params
        super().__init__(features_dir)

    def fit(self, df):
        self.vectorizer = TextVectorizerFactory().run(model_str=self.vectorizer_str, params=self.vectorizer_params)
        self.vectorizer.fit(df[self.input_col])
        return self

    def transform(self, df):
        vectors = self.vectorizer.transform(df[self.input_col])
        return pd.DataFrame(
            vectors,
            columns=[f"{self.input_col}_{self.vectorizer.return_flag()}_{i+1}" for i in range(vectors.shape[1])],
        )


class DecomposedSentenceVector(FeaturesPipeline):
    """BertやWordCountのベクトルをPCAやUMAP,NMF,SVDなどで次元削減する.次元削減前の標準化するかどうかも選べる."""

    def __init__(
        self,
        input_col,
        vectorizer_str,
        decomposer_str,
        n_components,
        vectorizer_params=None,
        standardize=False,
        seed=None,
        features_dir=None,
    ):
        vectorizer = SentenceVectorFeature(
            input_col=input_col,
            vectorizer_str=vectorizer_str,
            vectorizer_params=vectorizer_params,
            # これを忘れてしまう...
            features_dir=features_dir,
        )

        if standardize:
            decomposer = StandardizedDecomposer
        else:
            decomposer = DecompositionFeatures

        # TODO : ここ本当は一意にしたいけど長くなってしまう...
        prefix = f"{input_col}_{vectorizer_str}_{decomposer_str}_{n_components}"

        decomposer = decomposer(
            decomposer_str=decomposer_str,
            n_components=n_components,
            seed=seed,
            features_dir=features_dir,
            prefix=prefix,
        )

        super().__init__(feature_generators=[vectorizer, decomposer], use_prev_only=True, features_dir=features_dir)
