import pandas as pd
from sklearn.decomposition import PCA
from umap import UMAP

from lilac.features.generator_base import FeaturesBase


class DecompositionFeatures(FeaturesBase):
    def __init__(
        self, decomposer_str, n_components, input_cols=None, col_mark=None, random_state=None, features_dir=None
    ):
        """PCAやUMAPを用いて次元削減する.

        :n_components: 削減先の次元数.
        :input_cols: 適用するカラム.指定しないと全カラムになる.
        :col_mark: カラム名が{pca}_{col_mark}_1,...となる.指定しないとinput_colsの先頭のcolumn名=col_markになる
        :random_state
        :features_dir
        """
        self.decomposer_str = decomposer_str
        self.input_cols = input_cols
        self.n_components = n_components
        self.random_state = random_state
        self.col_mark = col_mark
        super().__init__(features_dir)

    def fit(self, df):
        models = {"pca": PCA, "umap": UMAP}
        input_cols = self.resolve_input_cols(df)
        self.model = models[self.decomposer_str](n_components=self.n_components, random_state=self.random_state)
        self.model.fit(df[input_cols])
        return self

    def transform(self, df):
        input_cols = self.resolve_input_cols(df)
        data = self.model.transform(df[input_cols])
        if self.col_mark is None:
            col_mark = input_cols[0]
        else:
            col_mark = self.col_mark
        output_cols = [f"{self.decomposer_str}_{col_mark}_{i+1}" for i in range(self.n_components)]
        return pd.DataFrame(data, columns=output_cols)

    def resolve_input_cols(self, df):
        if self.input_cols is None:
            return df.columns
        else:
            return self.input_cols
