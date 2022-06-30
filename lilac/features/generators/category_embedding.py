import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

from lilac.features.generator_base import FeaturesBase
from lilac.features.generators.combination_features import CategoryCombination
from lilac.features.wrappers.features_pipeline import FeaturesPipeline


class ConcatCategoriesLda(FeaturesPipeline):
    """カテゴリを結合してLdaのsub_colに指定する."""

    def __init__(self, num_topics, main_col, sub_cols, output_suffix="_cc", features_dir=None):
        concatter = CategoryCombination(input_cols=sub_cols, output_suffix=output_suffix, features_dir=features_dir)
        sub_col = "".join(sub_cols) + output_suffix
        vectorizer = CategoriesLdaVectorizer(
            num_topics=num_topics, main_col=main_col, sub_col=sub_col, features_dir=features_dir
        )
        super().__init__(feature_generators=[concatter, vectorizer], features_dir=features_dir)


class CategoriesLdaVectorizer(FeaturesBase):
    """main_colでgroupbyしてsub_colを連結したものを文章と見做してLDAでベクトルに変換する.

    main_colの方が種類が少なくて、同じsub_colが複数種類のmain_colに出現するような時に使える.
    """

    def __init__(self, num_topics, main_col, sub_col, features_dir=None):
        self.num_topics = num_topics
        self.main_col = main_col
        self.sub_col = sub_col
        super().__init__(features_dir=features_dir)

    def fit(self, df):
        def concat_values(group):
            values = group.fillna("[NONE]").values.tolist()
            return " ".join(values)

        docs_df = df.groupby(self.main_col)[self.sub_col].apply(concat_values)
        docs = docs_df.values
        keys = docs_df.keys()
        docs = [doc.split(" ") for doc in docs]
        data = LdaVectorizer(num_topics=self.num_topics).fit_transform(docs)
        lda_df = pd.DataFrame(
            data, columns=[f"{self.main_col}_{self.sub_col}_lda_{self.num_topics}-{i}" for i in range(data.shape[1])]
        )
        lda_df[self.main_col] = keys
        self.lda_df = lda_df
        return self

    def transform(self, df):
        merged = pd.merge(df, self.lda_df, on=self.main_col)
        return merged[self.lda_df.columns].drop(self.main_col, axis=1)


class LdaVectorizer:
    def __init__(self, num_topics):
        self.num_topics = num_topics

    def fit(self, docs):
        corpus = self.get_corpus(docs)
        # トピック数を指定してモデルを学習
        self.lda = LdaModel(corpus, num_topics=self.num_topics)
        return self

    def get_corpus(self, docs):
        # 単語と単語IDを対応させる辞書の作成
        dictionary = Dictionary(docs)
        # LdaModelが読み込めるBoW形式に変換
        return [dictionary.doc2bow(text) for text in docs]

    def transform(self, docs):
        corpus = self.get_corpus(docs)
        topic_df = pd.DataFrame(index=range(len(corpus)))
        for c in range(self.num_topics):
            topic_df[c] = 0.0
        for i in range(len(corpus)):
            topics = self.lda[corpus[i]]
            for t, p in topics:
                topic_df.loc[i][t] = p
        return topic_df.values

    def fit_transform(self, docs):
        return self.fit(docs).transform(docs)
