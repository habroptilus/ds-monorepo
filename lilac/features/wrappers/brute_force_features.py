import itertools

import pandas as pd

from lilac.features.generator_base import _FeaturesBase


class BruteForceFeatures(_FeaturesBase):
    """複数のパラメータについて複数候補を渡したときに、全組み合わせで特徴量生成クラスを内側で呼ぶ.

    保存は内側のみ(一つずつ)行われる.
    TODO もう少し汎用的にかけるかも(二種類の引数に限定しない書き方)
    """

    def __init__(self, params_a, params_b, name_a, name_b, FeaturesClass, **kwargs):
        self.FeaturesClass = FeaturesClass
        self.features_generators = []
        for param_a, param_b in itertools.product(params_a, params_b):
            params = {name_a: param_a, name_b: param_b}
            self.features_generators.append(FeaturesClass(**params, **kwargs))

    def run(self, train, test):
        train_res_list = []
        test_res_list = []
        for gen in self.features_generators:
            train_res, test_res = gen.run(train, test)
            train_res_list.append(train_res)
            test_res_list.append(test_res)
        train_res = pd.concat(train_res_list, axis=1)
        test_res = pd.concat(test_res_list, axis=1)
        return train_res, test_res

    def fit(self, df):
        for gen in self.features_generators:
            # 参照渡しなはず...
            gen.fit(df)
        return self

    def transform(self, df):
        features = []
        for gen in self.features_generators:
            feature = gen.transform(df)

            features.append(feature)
        return pd.concat(features, axis=1)
