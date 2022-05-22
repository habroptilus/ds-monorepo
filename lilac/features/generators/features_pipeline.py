import pandas as pd
from lilac.features.generator_base import FeaturesBase, _FeaturesBase


class PipelineBlock(FeaturesBase):
    """generatorを二つ受け取って、ひとつ目に依存した二つ目のgeneratorとして振る舞う."""

    def __init__(self, generator_first, generator_second, use_prev_only=False, features_dir=None):
        self.generator_first = generator_first
        self.generator_second = generator_second
        self.use_prev_only = use_prev_only
        super().__init__(features_dir)

    def _generate(self, train_data, test_data):
        """_generateをオーバーライドしないと,子genereatorのrunが呼ばれないので子のsave,load機能が動作しないのでこれは必須."""
        train_data = train_data.copy()
        test_data = test_data.copy()

        # 依存先のgeneratorはrunを呼ぶことでsave,load機能を用いる
        train_features, test_features = self.generator_first.run(
            train_data, test_data)
        train_data = self.concat_or_not(
            train_data, train_features)
        test_data = self.concat_or_not(
            test_data, test_features)

        # 本体は保存せずPipelineBlockの保存で保存されるようにする.
        self.generator_second.fit(
            pd.concat([train_data, test_data]).reset_index(drop=True))
        train_features = self.generator_second.transform(train_data)
        test_features = self.generator_second.transform(test_data)

        return train_features, test_features

    def fit(self, df):
        new_features = self.generator_first.fit_transform(df)
        df = self.concat_or_not(df, new_features)
        self.generator_second.fit(df)
        return self

    def transform(self, df):
        new_features = self.generator_first.transform(df)
        df = self.concat_or_not(df, new_features)
        return self.generator_second.transform(df)

    def calc_md5(self):
        return self._calc_md5({"first": self.generator_first.md5, "second": self.generator_second.md5, "use_prev_only": self.use_prev_only})

    def return_flag(self):
        return f"{self.generator_second.__class__.__name__}_by_pipeline_{self.md5}"

    def concat_or_not(self, original, diff):
        original = original.copy()
        diff = diff.copy()
        if self.use_prev_only:
            return diff
        else:
            return pd.concat([original, diff], axis=1)


class FeaturesPipeline(_FeaturesBase):
    """FeaturesBaseを複数直列に並べてパイプラインを作る."""

    def __init__(self, feature_generators, use_prev_only=False, features_dir=None):
        """use_previous_cols=Trueにするとひとつ前のgeneratorの出力を次のinputとする.

        :use_previous_cols Bool or List[Bool]
        Listを指定する場合はfeature_generatorsの個数よりひとつ少ないものを指定する必要がある.
        boolの場合は長さfeature_generators-1のリストにキャストされる.
        """
        if isinstance(use_prev_only, bool):
            use_prev_only_list = [use_prev_only] * \
                (len(feature_generators)-1)
        elif isinstance(use_prev_only, list):
            if len(use_prev_only) != len(feature_generators) - 1:
                raise Exception(
                    f"Length of use_previous_cols unmatched { len(use_prev_only)} != {len(feature_generators)- 1}")
            use_prev_only_list = use_prev_only
        self.use_prev_only_list = use_prev_only_list
        self.features_dir = features_dir
        self.main_block = self.get_nested_block(feature_generators)

    def get_nested_block(self, feature_generators):
        prev = None
        for i, gen in enumerate(feature_generators):
            if prev is None:
                prev = gen
                continue
            prev = PipelineBlock(generator_first=prev, generator_second=gen,
                                 use_prev_only=self.use_prev_only_list[i-1], features_dir=self.features_dir)
        return prev

    def run(self, train, test):
        return self.main_block.run(train, test)

    def fit(self, df):
        self.main_block.fit_transform(df)
        return self

    def transform(self, df):
        return self.main_block.transform(df)

    def return_flag(self):
        return "pipeline"
