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
    """特徴量生成器のFactoryクラス."""

    def __init__(self, features_dir, register_from=None):
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
        shared_params = {"features_dir": features_dir}
        super().__init__(str2model, register_from, shared_params)
