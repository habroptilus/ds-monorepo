import pandas as pd
from sklearn import cluster, mixture

from lilac.features.generator_base import FeaturesBase


class ClusteringFeatures(FeaturesBase):
    """GMMやKmeansを用いてクラスタリングする. 出力はデフォルトでは所属クラスタを返す.

    fit時、入力がuniqueになるように前処理してから学習する(同じデータ点が複数あると重みがかかってしまうため)
    :params
    :model_str: クラスタリングアルゴリズム. kmeans, gmmから選択できる.
    :include_additional: kmeansなら各セントロイドへの距離、gmmなら各クラスタへの所属確率を特徴量に含むか
    """

    def __init__(
        self, model_str, input_cols, n_clusters, include_additional=False, random_state=None, features_dir=None
    ):
        self.input_cols = input_cols
        self.model_str = model_str
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.include_additional = include_additional
        super().__init__(features_dir)

    def fit(self, df):
        if self.model_str == "kmeans":
            self.model = cluster.KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        elif self.model_str == "gmm":
            self.model = mixture.GaussianMixture(
                n_components=self.n_clusters, covariance_type="full", random_state=self.random_state
            )
        else:
            raise Exception(f"Invalid model_str '{self.model_str}'")
        df = df[self.input_cols].drop_duplicates()
        self.model.fit(df[self.input_cols])
        return self

    def transform(self, df):
        result_df = pd.DataFrame()
        if self.include_additional:
            if self.model_str == "kmeans":
                data = self.model.transform(df[self.input_cols])
            elif self.model_str == "gmm":
                data = self.model.predict_proba(df[self.input_cols])
            else:
                raise Exception(f"Invalid model_str '{self.model_str}'")
            cols = [f"{self.model_str}_{'_'.join(self.input_cols)}_{i}" for i in range(self.n_clusters)]
            result_df = pd.DataFrame(data, columns=cols)

        clusters = self.model.predict(df[self.input_cols])

        result_df[f"{self.model_str}_{'_'.join(self.input_cols)}_{self.n_clusters}_cluster"] = [
            f"cluster{cluster}" for cluster in clusters
        ]
        return result_df
