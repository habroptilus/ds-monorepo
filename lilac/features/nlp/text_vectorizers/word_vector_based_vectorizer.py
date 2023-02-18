import numpy as np

from lilac.features.nlp.text_vectorizers.text_vectorizer_base import TextVectorizerBase
from lilac.features.nlp.word_vectorizers.word_vector_factory import WordVectorizerFactory


class WordVectorBasedVectorizer(TextVectorizerBase):
    """WordVectorを集約してTextをVectorizeする."""

    def __init__(self, word_vectorizer_str, word_vector_size, how_to_aggregate="mean", seed=None, sep=" "):
        self.sep = sep
        self.vector_size = word_vector_size
        self.how_to_aggregate = how_to_aggregate
        self.word_vectorizer_str = word_vectorizer_str
        params = {"vector_size": word_vector_size, "seed": seed, "sep": self.sep}
        self.word_vectorizer = WordVectorizerFactory().run(model_str=word_vectorizer_str, params=params)

    def fit(self, docs):
        """docs: 文字列のリスト"""
        self.word_vectorizer.fit(docs)
        return self

    def transform(self, docs):
        """fitと同じ入力を想定"""
        data = []
        for doc in docs:
            words = doc.split(self.sep)
            vectors = []
            for word in words:
                vector = self.word_vectorizer.transform(word)
                if vector is not None:
                    # unknown wordでなければvectorを採用
                    vectors.append(vector)
            if len(vectors) == 0:
                # 一単語もベクトルにできなかった場合は0ベクトルを返す.
                mean_vec = np.zeros(self.vector_size)
            elif self.how_to_aggregate == "mean":
                mean_vec = np.array(vectors).mean(axis=0)
            else:
                raise Exception(f"Invalid aggregation option: {self.how_to_aggregate}")
            data.append(mean_vec)
        return np.array(data)

    def return_flag(self):
        return f"{self.word_vectorizer_str}_{self.how_to_aggregate}"

    def save(self, path):
        self.word_vectorizer.save(path)

    def load(self, path):
        self.word_vectorizer.load(path)
