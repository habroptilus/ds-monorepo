from tools.core.factory_base import FactoryBase
from tools.features.generators.category_encoding import CategoryEncoding
from tools.features.generators.datetime_features import DatetimeFeatures
from tools.features.generators.sentence_vectorizer import WordCountVectorizer, BertVectorizer
from tools.features.generators.extra_table import ExtraTableJoin
from tools.features.generators.clustering_features import ClusteringFeatures
from tools.features.generators.features_pipeline import FeaturesPipeline
from tools.features.generators.decomposer_features import DecompositionFeatures
from tools.features.generators.group_features import GroupFeatures


class FeatureGeneratorsFactory(FactoryBase):

    def __init__(self, features_dir, custom_members=None):
        str2model = {
            "category": CategoryEncoding,
            "datetime": DatetimeFeatures,
            "wc_vec": WordCountVectorizer,
            "extra_join": ExtraTableJoin,
            "cluster": ClusteringFeatures,
            "pipeline": FeaturesPipeline,
            "bert": BertVectorizer,
            "dec": DecompositionFeatures,
            "group": GroupFeatures
        }
        self.features_dir = features_dir
        super().__init__(str2model, custom_members)

    def get_params(self, params=None):
        if params is None:
            params = {}
        params["features_dir"] = self.features_dir
        return params

    def run(self, model_str, params):
        return super().run(model_str, params)
