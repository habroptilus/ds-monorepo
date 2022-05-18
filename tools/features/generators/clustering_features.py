import pandas as pd
import numpy as np
from tools.features.generator_base import FeaturesBase
from sklearn import cluster, mixture


class ClusteringFeatures(FeaturesBase):
    """GMMやKmeansを用いてクラスタリングする.出力はgmmは各クラスタへの所属確率と所属クラスタ.
    kmeansはセントロイドへの距離と所属クラスタ.
    """

    def __init__(self, model_str, input_cols,  n_clusters, random_state=None, features_dir=None):
        self.input_cols = input_cols
        self.model_str = model_str
        self.n_clusters = n_clusters
        self.random_state = random_state
        super().__init__(features_dir)

    def fit(self, df):
        if self.model_str == "kmeans":
            self.model = cluster.KMeans(
                n_clusters=self.n_clusters, random_state=self.random_state)
        elif self.model_str == "gmm":
            self.model = mixture.GaussianMixture(
                n_components=self.n_clusters, covariance_type='full', random_state=self.random_state)
        else:
            raise Exception(f"Invalid model_str '{self.model_str}'")
        self.model.fit(df[self.input_cols])
        return self

    def transform(self, df):
        if self.model_str == "kmeans":
            data = self.model.transform(df[self.input_cols])
            clusters = self.model.predict(df[self.input_cols])
        elif self.model_str == "gmm":
            data = self.model.predict_proba(df[self.input_cols])
            clusters = self.model.predict(df[self.input_cols])
        else:
            raise Exception(f"Invalid model_str '{self.model_str}'")
        cols = [
            f"{self.model_str}_{'_'.join(self.input_cols)}_{i}" for i in range(self.n_clusters)]
        df = pd.DataFrame(data, columns=cols)
        df[f"{self.model_str}_{'_'.join(self.input_cols)}_cluster"] = clusters
        return df
