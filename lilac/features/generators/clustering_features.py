import pandas as pd
from sklearn import cluster, mixture

from lilac.features.generator_base import FeaturesBase
from lilac.features.generators.features_pipeline import FeaturesPipeline
from lilac.features.generators.scaling_features import StandardScalingFeatures


class StandardizedClustering(FeaturesPipeline):
    """標準化をしてからGMMやKmeansを用いてクラスタリングを行う."""

    def __init__(
        self, model_str, input_cols, n_clusters, prefix, include_additional=False, random_state=None, features_dir=None
    ):
        super().__init__(
            feature_generators=[
                StandardScalingFeatures(input_cols=input_cols, features_dir=features_dir),
                ClusteringFeatures(
                    model_str=model_str,
                    n_clusters=n_clusters,
                    include_additional=include_additional,
                    random_state=random_state,
                    features_dir=features_dir,
                    prefix=prefix,
                ),
            ],
            use_prev_only=True,
            features_dir=features_dir,
        )


class ClusteringFeatures(FeaturesBase):
    """GMMやKmeansを用いてクラスタリングする. 出力はデフォルトでは所属クラスタを返す.

    fit時、入力がuniqueになるように前処理してから学習する(同じデータ点が複数あると重みがかかってしまうため)
    :params
    :model_str: クラスタリングアルゴリズム. kmeans, gmmから選択できる.
    :include_additional: kmeansなら各セントロイドへの距離、gmmなら各クラスタへの所属確率を特徴量に含むか
    :input_cols: クラスタリング対象の列. 指定しないと入力全体を対象とする.
    """

    def __init__(
        self,
        model_str,
        n_clusters,
        prefix,
        input_cols=None,
        include_additional=False,
        random_state=None,
        features_dir=None,
    ):
        self.input_cols = input_cols
        self.model_str = model_str
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.include_additional = include_additional
        self.prefix = prefix
        super().__init__(features_dir)

    def fit(self, df):
        self.input_cols = self.resolve_input_cols(df)
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
        key_base = f"{self.prefix}_{self.model_str}_{self.n_clusters}"
        if self.include_additional:
            if self.model_str == "kmeans":
                data = self.model.transform(df[self.input_cols])
            elif self.model_str == "gmm":
                data = self.model.predict_proba(df[self.input_cols])
            else:
                raise Exception(f"Invalid model_str '{self.model_str}'")
            cols = [f"{key_base}-{i}" for i in range(self.n_clusters)]
            result_df = pd.DataFrame(data, columns=cols)

        clusters = self.model.predict(df[self.input_cols])

        result_df[f"{key_base}_cluster"] = [f"cluster{cluster}" for cluster in clusters]
        return result_df

    def resolve_input_cols(self, df):
        if self.input_cols is None:
            return df.columns
        else:
            return self.input_cols
