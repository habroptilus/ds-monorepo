import hashlib
import json
from abc import ABCMeta, abstractmethod
from pathlib import Path

import pandas as pd

from lilac.core.utils import df_copy


class _FeaturesBase(metaclass=ABCMeta):
    def fit(self, df):
        return self

    @abstractmethod
    def transform(self, df):
        raise Exception("Not implemented error.")

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    @abstractmethod
    def return_flag(self):
        raise Exception("Not implemented error.")

    def __new__(cls, *args, **kwargs):
        """デコレーターを継承先にも適用させるため、インスタンス生成時にデコレーターを作用させるようにする."""
        cls.fit = df_copy(cls.fit)
        cls.transform = df_copy(cls.transform)
        return super().__new__(cls)


class FeaturesBase(_FeaturesBase):
    """特徴量生成の基底クラス.以下の機能を提供し、実際の特徴量計算は実装クラスで行う.

    :計算結果をsaveする.計算済みの特徴量がある場合はそれをloadする.
    """

    def __init__(self, features_dir=None):
        """features_dirを指定しない場合、save,load機能がオフになる.

        :継承先のinitでselfに登録したものを使ってハッシュ値を計算するので、encoderなどをインスタンス化したものがあると同一と見做されなくなる恐れあり
        """
        self.md5 = self.calc_md5()
        self.features_dir = Path(features_dir) if features_dir is not None else None

    def calc_md5(self):
        return self._calc_md5(vars(self))

    def _calc_md5(self, params):
        return hashlib.md5(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()

    def run(self, train_data, test_data):
        train = train_data.copy()
        test = test_data.copy()

        res_train, res_test = self._load()

        if res_train is not None:
            return res_train, res_test
        print(f"Generating {self.return_flag()}...")
        res_train, res_test = self._generate(train, test)

        self._save(res_train, res_test)

        return res_train, res_test

    def _save(self, res_train, res_test):
        if self.features_dir:
            train_path = self.features_dir / self.return_flag() / "train.ftr"
            train_path.parent.mkdir(exist_ok=True)
            res_train.to_feather(train_path)
            test_path = self.features_dir / self.return_flag() / "test.ftr"
            res_test.to_feather(test_path)

    def _load(self):
        res_train = None
        res_test = None
        if self.features_dir:
            train_path = self.features_dir / self.return_flag() / "train.ftr"
            test_path = self.features_dir / self.return_flag() / "test.ftr"
            if train_path.exists():
                # あるなら読み込む
                print(f"Loading {self.return_flag()} (train)...")
                res_train = pd.read_feather(train_path)
            if test_path.exists():
                print(f"Loading {self.return_flag()} (test)...")
                res_test = pd.read_feather(test_path)

        return res_train, res_test

    def _generate(self, train, test):
        # generatorのfitはtrain+testで行うことにする.
        # testのtarget_colはNoneが入るはず
        self.fit(pd.concat([test, train]).reset_index(drop=True))
        res_train = self.transform(train)
        res_test = self.transform(test)
        return res_train, res_test

    def return_flag(self):
        return f"{self.__class__.__name__}_{self.md5}"
