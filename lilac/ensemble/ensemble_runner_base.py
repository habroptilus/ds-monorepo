import pandas as pd
import numpy as np
from abc import ABCMeta, abstractmethod
from lilac.core.blocks import ModelingBlock


class _EnsembleRunnerBase(metaclass=ABCMeta):
    """アンサンブルを行う基底クラス.継承して実装する."""

    def __init__(self, target_col, unused_cols, folds_gen_factory_settings, model_factory_settings,
                 trainer_factory_settings, evaluator_flag, use_original_cols=False):
        """
        use_original_colsがFalseだとunused_colsはtestのカラムで置き換えられる
        """
        self.target_col = target_col
        self.use_original_cols = use_original_cols
        self.unused_cols = unused_cols
        self.folds_gen_factory_settings = folds_gen_factory_settings
        self.model_factory_settings = model_factory_settings
        self.trainer_factory_settings = trainer_factory_settings
        self.evaluator_flag = evaluator_flag

    def run(self, output_list, train, test):
        """前の層の予測からデータセットを作り、ModelingBlockに渡して出力を得る."""
        # 前の層の予測からデータセットを作る
        # 目的変数は含まれていない
        output_based_train, output_based_test = self._create_datasets(
            output_list)

        # とりあえず全部のデータを結合する
        # foldの計算に必要だったりするため.
        train_df = pd.concat([output_based_train, train], axis=1)
        test_df = pd.concat([output_based_test, test], axis=1)

        if not self.use_original_cols:
            # 元のカラムを使わない時はそれをunused colsにする
            unused_cols = list(test.columns)
        else:
            unused_cols = self.unused_cols

        modeling_block = ModelingBlock(
            self.target_col, unused_cols, self.folds_gen_factory_settings, self.model_factory_settings, self.trainer_factory_settings, self.evaluator_flag)

        return modeling_block.run(train_df, test_df)

    @abstractmethod
    def _create_datasets(self, output_list):
        """一層前のoutput_listから予測値を使って特徴量を作る."""
        raise Exception("Implement please.")


class EnsembleRunnerBase(_EnsembleRunnerBase):
    """回帰、binary, multiクラス分類用アンサンブル基底クラス."""

    def _create_datasets(self, output_list):
        """一層前のoutput_listから予測値を使って特徴量を作る.

        oof_raw_pred -> train用
        raw_pred -> test用
        raw_predは回帰だと1次元、分類だと各クラスの予測確率なのでクラス数の次元である
        最終出力の次元は「一層前のmodel数*raw_predの次元数」となる.
        """
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for i, output in enumerate(output_list):
            oof_raw_pred = np.array(output["oof_raw_pred"])
            raw_pred = np.array(output["raw_pred"])
            if len(oof_raw_pred.shape) == 1:
                raw_pred = np.expand_dims(raw_pred, 1)
                oof_raw_pred = np.expand_dims(oof_raw_pred, -1)

            oof_raw_pred = pd.DataFrame(oof_raw_pred, columns=[
                                        f"pred{i}_{j}" for j in range(oof_raw_pred.shape[1])])

            raw_pred = pd.DataFrame(
                raw_pred, columns=[f"pred{i}_{j}" for j in range(raw_pred.shape[1])])
            train_df = pd.concat([train_df, oof_raw_pred], axis=1)
            test_df = pd.concat([test_df, raw_pred], axis=1)
        return train_df, test_df


class LrRmsleEnsembleRunnerBase(_EnsembleRunnerBase):
    """入力のlogを取る.RMSLEで線形モデルを使う場合に用いる."""

    def _create_datasets(self, output_list):
        """予測値をlog(x+1)で変換する.回帰のみ対応している"""
        train_df = pd.DataFrame()
        test_df = pd.DataFrame()
        for i, output in enumerate(output_list):
            # 予測値の対数をとって入力にする(linearだと効果あるはず)
            train_df[f"pred{i}"] = np.log1p(output["oof_raw_pred"])
            test_df[f"pred{i}"] = np.log1p(output["raw_pred"])
        return train_df, test_df
