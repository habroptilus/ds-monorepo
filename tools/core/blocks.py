from tools.evaluators.evaluator_factory import EvaluatorFactory
from tools.models.model_factory import ModelFactory
from tools.trainers.trainer_factory import TrainerFactory
from tools.validators.cross_validation_runner import CrossValidationRunner
from tools.validators.folds_generator_factory import FoldsGeneratorFactory
from tools.features.generator_factory import FeatureGeneratorsFactory
from tools.features.features_aggregator import FeaturesAggregator


class ModelingBlock:
    def __init__(self, target_col, unused_cols, folds_gen_factory_settings, model_factory_settings, trainer_factory_settings, evaluator_flag):
        self.folds_generator = FoldsGeneratorFactory().run(**folds_gen_factory_settings)
        self.unused_cols = unused_cols
        trainer = TrainerFactory().run(**trainer_factory_settings)
        model_factory = ModelFactory(target_col)
        evaluator = EvaluatorFactory().run(evaluator_flag)
        self.cv_runner = CrossValidationRunner(pred_oof=True, target_col=target_col, model_factory=model_factory,
                                               model_params=model_factory_settings, trainer=trainer, evaluator=evaluator)

    def run(self, train, test):
        folds = self.folds_generator.run(train)
        train = train.drop(self.unused_cols, axis=1)
        test = test.drop(self.unused_cols, axis=1)
        output = self.cv_runner.run(train, folds)
        output["raw_pred"] = list(self.cv_runner.raw_output(test))
        output["pred"] = list(self.cv_runner.final_output(test))
        return output


class DatagenBlock:
    def __init__(self, target_col, features_dir, custom_members, features_settings):
        features_factory = FeatureGeneratorsFactory(
            features_dir=features_dir, custom_members=custom_members)
        self.aggregator = FeaturesAggregator(
            target_col=target_col, factory=features_factory, settings=features_settings)

    def run(self, train, test):
        return self.aggregator.run(train, test)


class BlocksRunner:
    def __init__(self, features_dir, custom_members, features_settings, target_col, unused_cols, folds_gen_factory_settings, model_factory_settings, trainer_factory_settings,  evaluator_flag):
        self.datagen_block = DatagenBlock(
            target_col, features_dir, custom_members, features_settings)
        self.modeling_block = ModelingBlock(
            target_col, unused_cols, folds_gen_factory_settings, model_factory_settings, trainer_factory_settings, evaluator_flag)

    def run(self, train, test):
        train, test = self.datagen_block.run(train, test)
        return self.modeling_block.run(train, test)
