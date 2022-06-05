from typing import List, Optional

import pandas as pd

from lilac.evaluators.evaluator_factory import EvaluatorFactory
from lilac.features.features_aggregator import FeaturesAggregator
from lilac.features.generator_factory import FeatureGeneratorsFactory
from lilac.features.target_encoders.target_encoder import TargetEncoder
from lilac.models.model_factory import ModelFactory
from lilac.trainers.trainer_factory import TrainerFactory
from lilac.validators.cross_validation_runner import CrossValidationRunner
from lilac.validators.folds_generator_factory import FoldsGeneratorFactory


class ModelingBlock:
    """学習データを受け取り、モデル学習と評価を行い結果をdictで返す."""

    def __init__(
        self,
        target_col,
        unused_cols,
        folds_gen_factory_settings,
        model_factory_settings,
        trainer_factory_settings,
        evaluator_str,
        log_target_on_target_enc=False,
        target_enc_cols=None,
        seed=None,
    ):
        folds_generator = FoldsGeneratorFactory().run(**folds_gen_factory_settings)
        trainer = TrainerFactory().run(**trainer_factory_settings)
        model_factory = ModelFactory(target_col)
        evaluator = EvaluatorFactory().run(evaluator_str)

        self.cv_runner = CrossValidationRunner(
            pred_oof=True,
            target_col=target_col,
            model_factory=model_factory,
            model_params=model_factory_settings,
            trainer=trainer,
            evaluator=evaluator,
            folds_generator=folds_generator,
            unused_cols=unused_cols,
            target_enc_cols=target_enc_cols,
            seed=seed,
            log_target_on_target_enc=log_target_on_target_enc,
        )

    def run(self, train, test):
        """target_encoding, run cv.

        Target encodingをした場合、元のカラムは削除される.
        """
        # run cross validation and return prediction
        output = self.cv_runner.run(train)
        predictions = self.cv_runner.get_predictions(test)
        output["raw_pred"] = predictions.raw_pred
        output["pred"] = predictions.pred
        return output


class DatagenBlock:
    """特徴量生成を行い、それらをまとめて学習/テストデータを作る."""

    def __init__(self, target_col, features_dir, register_from, extra_class_names: Optional[List], features_settings):
        features_factory = FeatureGeneratorsFactory(
            features_dir=features_dir, register_from=register_from, extra_class_names=extra_class_names
        )
        self.aggregator = FeaturesAggregator(
            target_col=target_col, factory=features_factory, settings=features_settings
        )

    def run(self, train, test):
        return self.aggregator.run(train, test)


class BlocksRunner:
    """学習データ作成からモデル生成、評価まで行う."""

    # TODO  register_fromは今はfeatures_genだけだが、modelとかもカスタムモデルを使えるようにしたい

    def __init__(
        self,
        features_dir,
        register_from,
        extra_class_names: Optional[List],
        features_settings,
        target_col,
        folds_gen_factory_settings,
        model_factory_settings,
        trainer_factory_settings,
        evaluator_str,
        log_target_on_target_enc=False,
        unused_cols=None,
        target_enc_cols=None,
        seed=None,
    ):
        self.datagen_block = DatagenBlock(
            target_col, features_dir, register_from, extra_class_names, features_settings
        )
        self.modeling_block = ModelingBlock(
            target_col=target_col,
            unused_cols=unused_cols,
            folds_gen_factory_settings=folds_gen_factory_settings,
            model_factory_settings=model_factory_settings,
            trainer_factory_settings=trainer_factory_settings,
            evaluator_str=evaluator_str,
            log_target_on_target_enc=log_target_on_target_enc,
            target_enc_cols=target_enc_cols,
            seed=seed,
        )

    def run(self, train, test):
        train, test = self.datagen_block.run(train, test)
        return self.modeling_block.run(train, test)
