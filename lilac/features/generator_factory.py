from lilac.core.factory_base import FactoryBase
from lilac.features.generators.category_embedding import CategoriesLdaVectorizer
from lilac.features.generators.category_encoding import CategoryEncoding
from lilac.features.generators.clustering_features import ClusteringFeatures, StandardizedClustering
from lilac.features.generators.combination_features import CategoryCombination, NumericCombination
from lilac.features.generators.datetime_features import DatetimeFeatures
from lilac.features.generators.decomposer_features import DecompositionFeatures, StandardizedDecomposer
from lilac.features.generators.extra_table import ExtraTableJoin
from lilac.features.generators.group_features import GroupFeatures
from lilac.features.generators.lag_features import LagFeatures
from lilac.features.generators.scaling_features import StandardScalingFeatures
from lilac.features.generators.sentence_vectorizer import (
    BertVectorizer,
    DecomposedSentenceVectoizer,
    WordCountVectorizer,
)


class FeatureGeneratorsFactory(FactoryBase):
    """特徴量生成器のFactoryクラス."""

    def __init__(self, features_dir, register_from=None, extra_class_names=None):
        str2model = {
            "category": CategoryEncoding,
            "datetime": DatetimeFeatures,
            "extra_join": ExtraTableJoin,
            "group": GroupFeatures,
            "lag": LagFeatures,
            "std_scale": StandardScalingFeatures,
            "dec": DecompositionFeatures,
            "std_dec": StandardizedDecomposer,
            "cluster": ClusteringFeatures,
            "std_cluster": StandardizedClustering,
            "sv_dec": DecomposedSentenceVectoizer,
            "word_count": WordCountVectorizer,
            "bert": BertVectorizer,
            "cat_combi": CategoryCombination,
            "num_combi": NumericCombination,
            "cat_lda": CategoriesLdaVectorizer,
        }
        shared_params = {"features_dir": features_dir}
        super().__init__(
            str2model=str2model,
            register_from=register_from,
            shared_params=shared_params,
            extra_class_names=extra_class_names,
        )
