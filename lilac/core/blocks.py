from typing import List, Optional

from lilac.evaluators.evaluator_factory import EvaluatorFactory
from lilac.features.features_aggregator import FeaturesAggregator
from lilac.features.generator_factory import FeatureGeneratorsFactory
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
    ):
        self.folds_generator = FoldsGeneratorFactory().run(**folds_gen_factory_settings)
        self.unused_cols = unused_cols
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
        )

    def run(self, train, test):
        folds = self.folds_generator.run(train)
        train = train.drop(self.unused_cols, axis=1)
        test = test.drop(self.unused_cols, axis=1)
        output = self.cv_runner.run(train, folds)
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
        unused_cols,
        folds_gen_factory_settings,
        model_factory_settings,
        trainer_factory_settings,
        evaluator_str,
    ):
        self.datagen_block = DatagenBlock(
            target_col, features_dir, register_from, extra_class_names, features_settings
        )
        self.modeling_block = ModelingBlock(
            target_col,
            unused_cols,
            folds_gen_factory_settings,
            model_factory_settings,
            trainer_factory_settings,
            evaluator_str,
        )

    def run(self, train, test):
        train, test = self.datagen_block.run(train, test)
        return self.modeling_block.run(train, test)
