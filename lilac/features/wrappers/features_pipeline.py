import pandas as pd

from lilac.features.generator_base import FeaturesBase, _FeaturesBase


class PipelineComponent(FeaturesBase):
    """generatorを二つ受け取って、ひとつ目に依存した二つ目のgeneratorとして振る舞う."""

    def __init__(self, generator_first, generator_second, use_prev_only=False, features_dir=None):
        self.generator_first = generator_first
        self.generator_second = generator_second
        self.use_prev_only = use_prev_only
        super().__init__(features_dir)

    def run(self, train, test):
        # 自分の出力があればloadする
        train_features, test_features = self._load()

        if train_features is not None:
            return train_features, test_features

        # ない場合
        print(f"Generating {self.return_flag()}...")

        # 一個目のgeneratorをrun. saveされる
        train_features, test_features = self.generator_first.run(train, test)

        train = self.concat_or_not(train, train_features)
        test = self.concat_or_not(test, test_features)

        # 二個目のgeneratorをfit_transform. saveされない
        train_features = self.generator_second.fit_transform(train)
        test_features = self.generator_second.fit_transform(test)

        # 自分自身の出力をsave
        self._save(train_features, test_features)

        return train_features, test_features

    def fit(self, df):
        """保存されないように子genのfit,transformを使う."""
        new_features = self.generator_first.fit_transform(df)
        df = self.concat_or_not(df, new_features)
        self.generator_second.fit(df)
        return self

    def transform(self, df):
        """保存されないように子genのfit,transformを使う."""
        new_features = self.generator_first.transform(df)
        df = self.concat_or_not(df, new_features)
        return self.generator_second.transform(df)

    def calc_md5(self):
        return self._calc_md5(
            {
                "first": self.generator_first.md5,
                "second": self.generator_second.md5,
                "use_prev_only": self.use_prev_only,
            }
        )

    def return_flag(self):
        return f"{self.generator_second.__class__.__name__}/{self.md5}"

    def concat_or_not(self, original, diff):
        original = original.copy()
        diff = diff.copy()
        if self.use_prev_only:
            return diff
        else:
            return pd.concat([original, diff], axis=1)


class FeaturesPipeline(_FeaturesBase):
    """FeaturesBaseを複数直列に並べてパイプラインを作る.

    保存されるのは内側の特徴量生成クラスたち.
    """

    def __init__(self, feature_generators, use_prev_only=False, features_dir=None):
        """use_previous_cols=Trueにするとひとつ前のgeneratorの出力を次のinputとする.

        :use_previous_cols Bool or List[Bool]
        Listを指定する場合はfeature_generatorsの個数よりひとつ少ないものを指定する必要がある.
        boolの場合は長さfeature_generators-1のリストにキャストされる.
        """
        if isinstance(use_prev_only, bool):
            use_prev_only_list = [use_prev_only] * (len(feature_generators) - 1)
        elif isinstance(use_prev_only, list):
            if len(use_prev_only) != len(feature_generators) - 1:
                raise Exception(
                    f"Length of use_previous_cols unmatched { len(use_prev_only)} != {len(feature_generators)- 1}"
                )
            use_prev_only_list = use_prev_only
        self.use_prev_only_list = use_prev_only_list
        self.features_dir = features_dir
        self.pipeline = self.get_nested_components(feature_generators)
        self.md5 = self.pipeline.md5

    def get_nested_components(self, feature_generators) -> PipelineComponent:
        """feature_generatorsを二つずつ再起的に取り出してPipelineComponentを作っていき、最後のPipelineComponentを返す."""
        second = None
        for i, first in enumerate(feature_generators[::-1]):
            if first.features_dir is None:
                raise Exception(f"{first} doesn't have features_dir.")
            if second is None:
                second = first
                continue
            second = PipelineComponent(
                generator_first=first,
                generator_second=second,
                use_prev_only=self.use_prev_only_list[-i],
                features_dir=self.features_dir,
            )
        return second

    def run(self, train, test):
        return self.pipeline.run(train, test)

    def fit(self, df):
        self.pipeline.fit(df)
        return self

    def transform(self, df):
        return self.pipeline.transform(df)
